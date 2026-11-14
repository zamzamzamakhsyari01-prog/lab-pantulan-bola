// Pelepas bola: motor DC gearbox + driver L298N lewat ESP32
// Perintah serial (9600 baud): '1' = buka 90 derajat, '0' = kembali.
const int EN  = 25;
const int IN1 = 26;
const int IN2 = 27;
const unsigned long waktu90  = 300;  // lama putar ~90 derajat (ms)
const unsigned long waktuRem = 50;   // lama pengereman (ms)
bool posisiBuka = false;

void putarMaju()   { digitalWrite(IN1, HIGH); digitalWrite(IN2, LOW);  digitalWrite(EN, HIGH); }
void putarMundur() { digitalWrite(IN1, LOW);  digitalWrite(IN2, HIGH); digitalWrite(EN, HIGH); }
void remAktif() {
  digitalWrite(IN1, LOW); digitalWrite(IN2, LOW); digitalWrite(EN, HIGH);
  delay(waktuRem);
  digitalWrite(EN, LOW);
}

void setup() {
  // level LOW dulu agar motor tidak bergerak saat ESP32 boot
  digitalWrite(EN, LOW); digitalWrite(IN1, LOW); digitalWrite(IN2, LOW);
  pinMode(EN, OUTPUT); pinMode(IN1, OUTPUT); pinMode(IN2, OUTPUT);
  digitalWrite(EN, LOW); digitalWrite(IN1, LOW); digitalWrite(IN2, LOW);
  Serial.begin(9600);
  Serial.println("Ketik 1 untuk membuka 90 derajat");
  Serial.println("Ketik 0 untuk kembali ke posisi semula");
}

void loop() {
  if (Serial.available() > 0) {
    char perintah = Serial.read();
    if (perintah == '\r' || perintah == '\n') return;
    if (perintah == '1' && !posisiBuka) {
      putarMaju(); delay(waktu90); remAktif();
      posisiBuka = true;
      Serial.println("Motor berhenti pada posisi buka");
    } else if (perintah == '0' && posisiBuka) {
      putarMundur(); delay(waktu90); remAktif();
      posisiBuka = false;
      Serial.println("Motor kembali ke posisi semula");
    }
  }
}
