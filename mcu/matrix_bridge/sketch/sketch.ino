#include <Arduino_RouterBridge.h>
#include <Arduino_LED_Matrix.h>
#include <vector>

Arduino_LED_Matrix matrix;
uint8_t frame[104] = {0};

void draw(std::vector<uint8_t> next) {
  if (next.size() != 104) return;
  for (size_t i = 0; i < 104; ++i) frame[i] = min(next[i], (uint8_t)7);
}

void setup() {
  matrix.begin();
  matrix.setGrayscaleBits(3);
  matrix.clear();
  Bridge.begin();
  Bridge.provide("draw", draw);
}

void loop() {
  matrix.draw(frame);
  delay(20);
}
