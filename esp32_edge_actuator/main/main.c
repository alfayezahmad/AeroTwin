#include "driver/gpio.h"
#include "esp_event.h"
#include "esp_log.h"
#include "esp_netif.h"
#include "esp_system.h"
#include "esp_wifi.h"
#include "freertos/FreeRTOS.h"
#include "freertos/queue.h"
#include "freertos/semphr.h"
#include "freertos/task.h"
#include "lwip/dns.h"
#include "lwip/netdb.h"
#include "lwip/sockets.h"
#include "mqtt_client.h"
#include "nvs_flash.h"
#include "led_strip.h"
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

static const char *TAG = "AEROTWIN_EDGE";

// ==========================================
// CONFIGURATION (CHANGE THESE FOR YOUR SETUP)
// ==========================================
#define WIFI_SSID "AlfayezHotspot"
#define WIFI_PASS "123456789"
#define MQTT_BROKER_URI                                                        \
  "mqtt://10.50.33.99" // Replace with your laptop's local IP address

// ==========================================
// PIN DEFINITIONS (ESP32-S3-N16R8 CAM Module)
// ==========================================
// GPIO 33 is the built-in red LED on many ESP32-S3 boards.
// If it's a specific CAM board, it might be 2 or 4. Change if needed.
#define LED_PIN 48

// L293D Motor Driver Pins (Direction control)
// Note: Ensure the L293D Enable pin is pulled HIGH to 5V/3.3V, or connect it to
// another GPIO and set it HIGH.
#define MOTOR_IN1_PIN 4
#define MOTOR_IN2_PIN 5

static led_strip_handle_t led_strip;

static void configure_gpios(void) {
  ESP_LOGI(TAG, "Configuring GPIOs...");

  // Configure NeoPixel
  led_strip_config_t strip_config = {
      .strip_gpio_num = LED_PIN,
      .max_leds = 1,
  };
  led_strip_rmt_config_t rmt_config = {
      .resolution_hz = 10 * 1000 * 1000, // 10MHz
  };
  ESP_ERROR_CHECK(led_strip_new_rmt_device(&strip_config, &rmt_config, &led_strip));
  led_strip_clear(led_strip);

  gpio_reset_pin(MOTOR_IN1_PIN);
  gpio_set_direction(MOTOR_IN1_PIN, GPIO_MODE_OUTPUT);
  gpio_set_level(MOTOR_IN1_PIN, 0); // Start OFF

  gpio_reset_pin(MOTOR_IN2_PIN);
  gpio_set_direction(MOTOR_IN2_PIN, GPIO_MODE_OUTPUT);
  gpio_set_level(MOTOR_IN2_PIN, 0); // Start OFF
}

// Non-blocking FreeRTOS Task to handle the 10-second mitigation
void trigger_mitigation_task(void *pvParameters) {
  ESP_LOGI(TAG, ">>> GRAP MITIGATION ENGAGED <<<");
  ESP_LOGI(TAG, "Turning ON onboard LED and Water Cannon Motor...");

  // Turn ON LED
  led_strip_set_pixel(led_strip, 0, 255, 0, 0); // Red
  led_strip_refresh(led_strip);

  // Turn ON Motor (Forward direction)
  gpio_set_level(MOTOR_IN1_PIN, 1);
  gpio_set_level(MOTOR_IN2_PIN, 0);

  // Wait 10 seconds exactly as requested
  vTaskDelay(10000 / portTICK_PERIOD_MS);

  ESP_LOGI(TAG, ">>> MITIGATION CYCLE COMPLETE <<<");
  ESP_LOGI(TAG, "Shutting down assets.");

  // Turn OFF LED
  led_strip_clear(led_strip);

  // Turn OFF Motor
  gpio_set_level(MOTOR_IN1_PIN, 0);
  gpio_set_level(MOTOR_IN2_PIN, 0);

  // Clean up task
  vTaskDelete(NULL);
}

static void mqtt_event_handler(void *handler_args, esp_event_base_t base,
                               int32_t event_id, void *event_data) {
  esp_mqtt_event_handle_t event = event_data;
  esp_mqtt_client_handle_t client = event->client;

  switch ((esp_mqtt_event_id_t)event_id) {
  case MQTT_EVENT_CONNECTED:
    ESP_LOGI(TAG, "MQTT Connected to Broker!");
    // Subscribe to the AeroTwin dispatch topic
    esp_mqtt_client_subscribe(client, "aerotwin/commands/dispatch", 0);
    break;

  case MQTT_EVENT_DATA:
    ESP_LOGI(TAG, "Received MQTT Command from AeroTwin Dashboard:");
    printf("TOPIC=%.*s\r\n", event->topic_len, event->topic);
    printf("DATA=%.*s\r\n", event->data_len, event->data);

    // We do a fast substring match to check if it's the spike simulator
    // triggering Stage III / IV
    if (strnstr(event->data, "WATER_SPRINKLER", event->data_len) != NULL ||
        strnstr(event->data, "Stage III", event->data_len) != NULL ||
        strnstr(event->data, "Stage IV", event->data_len) != NULL) {

      ESP_LOGW(
          TAG,
          "Pollution Spike Detected! Spinning up background actuation task.");
      // Spin up a FreeRTOS task so we don't block the MQTT receive thread
      xTaskCreate(trigger_mitigation_task, "mitigation_task", 2048, NULL, 5,
                  NULL);
    }
    break;

  case MQTT_EVENT_DISCONNECTED:
    ESP_LOGI(TAG, "MQTT Disconnected");
    break;

  default:
    break;
  }
}

static esp_mqtt_client_handle_t mqtt_client = NULL;

static void wifi_event_handler(void *arg, esp_event_base_t event_base,
                               int32_t event_id, void *event_data) {
  if (event_base == WIFI_EVENT && event_id == WIFI_EVENT_STA_START) {
    esp_wifi_connect();
  } else if (event_base == WIFI_EVENT &&
             event_id == WIFI_EVENT_STA_DISCONNECTED) {
    esp_wifi_connect();
    ESP_LOGW(TAG, "Retrying WiFi connection...");
  } else if (event_base == IP_EVENT && event_id == IP_EVENT_STA_GOT_IP) {
    ip_event_got_ip_t *event = (ip_event_got_ip_t *)event_data;
    ESP_LOGI(TAG, "Got IP: " IPSTR, IP2STR(&event->ip_info.ip));
    
    // Start MQTT only after we have an IP address
    if (mqtt_client == NULL) {
        ESP_LOGI(TAG, "Connecting to MQTT Broker at %s ...", MQTT_BROKER_URI);
        esp_mqtt_client_config_t mqtt_cfg = {
            .broker.address.uri = MQTT_BROKER_URI,
        };
        mqtt_client = esp_mqtt_client_init(&mqtt_cfg);
        esp_mqtt_client_register_event(mqtt_client, ESP_EVENT_ANY_ID, mqtt_event_handler, NULL);
        esp_mqtt_client_start(mqtt_client);
    }
  }
}

void wifi_init_sta(void) {
  ESP_ERROR_CHECK(esp_netif_init());
  ESP_ERROR_CHECK(esp_event_loop_create_default());
  esp_netif_create_default_wifi_sta();

  wifi_init_config_t cfg = WIFI_INIT_CONFIG_DEFAULT();
  ESP_ERROR_CHECK(esp_wifi_init(&cfg));

  ESP_ERROR_CHECK(esp_event_handler_instance_register(
      WIFI_EVENT, ESP_EVENT_ANY_ID, &wifi_event_handler, NULL, NULL));
  ESP_ERROR_CHECK(esp_event_handler_instance_register(
      IP_EVENT, IP_EVENT_STA_GOT_IP, &wifi_event_handler, NULL, NULL));

  wifi_config_t wifi_config = {
      .sta =
          {
              .ssid = WIFI_SSID,
              .password = WIFI_PASS,
          },
  };
  ESP_ERROR_CHECK(esp_wifi_set_mode(WIFI_MODE_STA));
  ESP_ERROR_CHECK(esp_wifi_set_config(WIFI_IF_STA, &wifi_config));
  ESP_ERROR_CHECK(esp_wifi_start());

  ESP_LOGI(TAG, "WiFi Station initialized.");
}

void app_main(void) {
  ESP_LOGI(TAG, "Starting AeroTwin Edge Actuator Node (ESP32-S3)");

  // Initialize NVS (Required for WiFi)
  esp_err_t ret = nvs_flash_init();
  if (ret == ESP_ERR_NVS_NO_FREE_PAGES ||
      ret == ESP_ERR_NVS_NEW_VERSION_FOUND) {
    ESP_ERROR_CHECK(nvs_flash_erase());
    ret = nvs_flash_init();
  }
  ESP_ERROR_CHECK(ret);

  // 1. Initialize Hardware Pins
  configure_gpios();

  // 2. Connect to WiFi
  wifi_init_sta();

  // The MQTT client is now automatically initialized inside the WiFi IP_EVENT_STA_GOT_IP handler.
}
