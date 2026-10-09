idf_component_register(SRCS "main.c"
                    INCLUDE_DIRS "."
                    REQUIRES mqtt nvs_flash esp_wifi esp_driver_gpio led_strip)
