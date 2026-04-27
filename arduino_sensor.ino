#include <Wire.h>
#include "MAX30105.h"   // SparkFun MAX3010x library — works for MAX30101

MAX30105 particleSensor;

const uint32_t CAPTURE_MS = 120000;   // 120 seconds (2 minutes)

void setup() {
  Serial.begin(115200);
  while (!Serial);

  Wire.begin();
  Wire.setClock(400000);

  if (!particleSensor.begin(Wire, I2C_SPEED_FAST)) {
    Serial.println("MAX30101 not found. Check wiring / 3.3V power.");
    while (1);
  }

  // setup(ledBrightness, sampleAverage, ledMode, sampleRate, pulseWidth, adcRange)
  //   ledMode    = 3  -> Red + IR + Green all active
  //   sampleRate = 100 Hz
  //   pulseWidth = 411 us  (18-bit resolution)
  //   adcRange   = 4096 nA
  particleSensor.setup(
      0x1F,   // default LED brightness (applied to Red)
      4,      // sample averaging
      3,      // LED mode: 3 = Red + IR + Green
      100,    // sample rate (Hz)
      411,    // pulse width (us)
      4096    // ADC range (nA)
  );

  // Tune each LED's brightness independently if you want
  particleSensor.setPulseAmplitudeRed(0x1F);
  particleSensor.setPulseAmplitudeIR(0x1F);
  particleSensor.setPulseAmplitudeGreen(0x1F);

  // CSV header
  Serial.println("time_ms,red,ir,green");
}

void loop() {
  static uint32_t startMs = millis();

  if (millis() - startMs >= CAPTURE_MS) {
    Serial.println("# capture complete");
    while (1);                        // stop here until reset
  }

  particleSensor.check();             // pull new samples into the library's FIFO

  while (particleSensor.available()) {
    uint32_t t     = millis() - startMs;
    uint32_t red   = particleSensor.getFIFORed();
    uint32_t ir    = particleSensor.getFIFOIR();
    uint32_t green = particleSensor.getFIFOGreen();

    Serial.print(t);     Serial.print(',');
    Serial.print(red);   Serial.print(',');
    Serial.print(ir);    Serial.print(',');
    Serial.println(green);

    particleSensor.nextSample();
  }
}
