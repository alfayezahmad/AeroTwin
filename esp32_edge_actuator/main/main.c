#include <stdio.h>
#include <stdint.h>
#include <stddef.h>
#include <string.h>
#include "esp_wifi.h"
#include "esp_system.h"
#include "nvs_flash.h"
#include "esp_event.h"
#include "esp_netif.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/semphr.h"
#include "freertos/queue.h"
#include "lwip/sockets.h"
#include "lwip/dns.h"
#include "lwip/netdb.h"
#include "esp_log.h"
#include "mqtt_client.h"
#include "driver/gpio.h"

static const char *TAG = "AEROTWIN_EDGE";

// ==========================================
// CONFIGURATION (CHANGE THESE FOR YOUR SETUP)
// ==========================================
#define WIFI_SSID      "YOUR_WIFI_SSID"
#define WIFI_PASS      "YOUR_WIFI_PASS"
#define MQTT_BROKER_URI "mqtt://192.168.1.XXX" // Replace with your laptop's local IP address

// ==========================================
// PIN DEFINITIONS (ESP32-S3-N16R8 CAM Module)
// ==========================================
// GPIO 33 is the built-in red LED on many ESP32-S3 boards. 
// If it's a specific CAM board, it might be 2 or 4. Change if needed.
#define LED_PIN        33 

// L293D Motor Driver Pins (Direction control)
// Note: Ensure the L293D Enable pin is pulled HIGH to 5V/3.3V, or connect it to another GPIO and set it HIGH.
#define MOTOR_IN1_PIN  4
#define MOTOR_IN2_PIN  5

static void configure_gpios(void)
{
    ESP_LOGI(TAG, "Configuring GPIOs...");
    
    gpio_reset_pin(LED_PIN);
    gpio_set_direction(LED_PIN, GPIO_MODE_OUTPUT);
    gpio_set_level(LED_PIN, 0); // Start OFF

    gpio_reset_pin(MOTOR_IN1_PIN);
    gpio_set_direction(MOTOR_IN1_PIN, GPIO_MODE_OUTPUT);
    gpio_set_level(MOTOR_IN1_PIN, 0); // Start OFF

    gpio_reset_pin(MOTOR_IN2_PIN);
    gpio_set_direction(MOTOR_IN2_PIN, GPIO_MODE_OUTPUT);
    gpio_set_level(MOTOR_IN2_PIN, 0); // Start OFF
}

// Non-blocking FreeRTOS Task to handle the 10-second mitigation
void trigger_mitigation_task(void *pvParameters)
{
    ESP_LOGI(TAG, ">>> GRAP MITIGATION ENGAGED <<<");
    ESP_LOGI(TAG, "Turning ON onboard LED and Water Cannon Motor...");
    
    // Turn ON LED
    gpio_set_level(LED_PIN, 1);
    
    // Turn ON Motor (Forward direction)
    gpio_set_level(MOTOR_IN1_PIN, 1);
    gpio_set_level(MOTOR_IN2_PIN, 0);

    // Wait 10 seconds exactly as requested
    vTaskDelay(10000 / portTICK_PERIOD_MS);

    ESP_LOGI(TAG, ">>> MITIGATION CYCLE COMPLETE <<<");
    ESP_LOGI(TAG, "Shutting down assets.");
    
    // Turn OFF LED
    gpio_set_level(LED_PIN, 0);
    
    // Turn OFF Motor
    gpio_set_level(MOTOR_IN1_PIN, 0);
    gpio_set_level(MOTOR_IN2_PIN, 0);

    // Clean up task
    vTaskDelete(NULL);
}

static void mqtt_event_handler(void *handler_args, esp_event_base_t base, int32_t event_id, void *event_data)
{
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
        
        // We do a fast substring match to check if it's the spike simulator triggering Stage III / IV
        if (strnstr(event->data, "WATER_SPRINKLER", event->data_len) != NULL || 
            strnstr(event->data, "Stage III", event->data_len) != NULL ||
            strnstr(event->data, "Stage IV", event->data_len) != NULL) {
            
            ESP_LOGW(TAG, "Pollution Spike Detected! Spinning up background actuation task.");
            // Spin up a FreeRTOS task so we don't block the MQTT receive thread
            xTaskCreate(trigger_mitigation_task, "mitigation_task", 2048, NULL, 5, NULL);
        }
        break;
        
    case MQTT_EVENT_DISCONNECTED:
        ESP_LOGI(TAG, "MQTT Disconnected");
        break;
        
    default:
        break;
    }
}

static void wifi_event_handler(void* arg, esp_event_base_t event_base, int32_t event_id, void* event_data)
{
    if (event_base == WIFI_EVENT && event_id == WIFI_EVENT_STA_START) {
        esp_wifi_connect();
    } else if (event_base == WIFI_EVENT && event_id == WIFI_EVENT_STA_DISCONNECTED) {
        esp_wifi_connect();
        ESP_LOGW(TAG, "Retrying WiFi connection...");
    } else if (event_base == IP_EVENT && event_id == IP_EVENT_STA_GOT_IP) {
        ip_event_got_ip_t* event = (ip_event_got_ip_t*) event_data;
        ESP_LOGI(TAG, "Got IP: " IPSTR, IP2STR(&event->ip_info.ip));
    }
}

void wifi_init_sta(void)
{
    ESP_ERROR_CHECK(esp_netif_init());
    ESP_ERROR_CHECK(esp_event_loop_create_default());
    esp_netif_create_default_wifi_sta();

    wifi_init_config_t cfg = WIFI_INIT_CONFIG_DEFAULT();
    ESP_ERROR_CHECK(esp_wifi_init(&cfg));

    ESP_ERROR_CHECK(esp_event_handler_instance_register(WIFI_EVENT, ESP_EVENT_ANY_ID, &wifi_event_handler, NULL, NULL));
    ESP_ERROR_CHECK(esp_event_handler_instance_register(IP_EVENT, IP_EVENT_STA_GOT_IP, &wifi_event_handler, NULL, NULL));

    wifi_config_t wifi_config = {
        .sta = {
            .ssid = WIFI_SSID,
            .password = WIFI_PASS,
        },
    };
    ESP_ERROR_CHECK(esp_wifi_set_mode(WIFI_MODE_STA));
    ESP_ERROR_CHECK(esp_wifi_set_config(WIFI_IF_STA, &wifi_config));
    ESP_ERROR_CHECK(esp_wifi_start());
    
    ESP_LOGI(TAG, "WiFi Station initialized.");
}

void app_main(void)
{
    ESP_LOGI(TAG, "Starting AeroTwin Edge Actuator Node (ESP32-S3)");

    // Initialize NVS (Required for WiFi)
    esp_err_t ret = nvs_flash_init();
    if (ret == ESP_ERR_NVS_NO_FREE_PAGES || ret == ESP_ERR_NVS_NEW_VERSION_FOUND) {
      ESP_ERROR_CHECK(nvs_flash_erase());
      ret = nvs_flash_init();
    }
    ESP_ERROR_CHECK(ret);

    // 1. Initialize Hardware Pins
    configure_gpios();

    // 2. Connect to WiFi
    wifi_init_sta();
    
    // Give WiFi a few seconds to get an IP before starting MQTT
    vTaskDelay(4000 / portTICK_PERIOD_MS);

    // 3. Connect to MQTT Broker
    ESP_LOGI(TAG, "Connecting to MQTT Broker at %s ...", MQTT_BROKER_URI);
    esp_mqtt_client_config_t mqtt_cfg = {
        .broker.address.uri = MQTT_BROKER_URI,
    };
    esp_mqtt_client_handle_t client = esp_mqtt_client_init(&mqtt_cfg);
    esp_mqtt_client_register_event(client, ESP_EVENT_ANY_ID, mqtt_event_handler, NULL);
    esp_mqtt_client_start(client);
}
