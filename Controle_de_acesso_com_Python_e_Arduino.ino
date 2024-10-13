int ledPin_on = 7;
int ledPin_off = 8;
int buzzerPin = 9;

void setup() {
  pinMode(ledPin_on, OUTPUT);
  pinMode(ledPin_off, OUTPUT);
  pinMode(buzzerPin, OUTPUT);
  
  Serial.begin(9600);  // Inicia comunicação serial
}

void loop() {
  if (Serial.available() > 0) {
    String command = Serial.readStringUntil('\n');  // Lê o comando recebido

    if (command == "Acesso_permitido") {
      digitalWrite(ledPin_on, HIGH);  // Acende o LED acesso permitido

      digitalWrite(buzzerPin, HIGH); //Ligar Buzzer
      delay(500); // Mantem o buzzer ligado por 1 segundo.
      digitalWrite(buzzerPin, LOW); //Desliga Buzzer
      
      delay(3000); // Mantem o LED de acesso permitido ligado por 3 segundos.
      digitalWrite(ledPin_on, LOW);  // Apaga o LED acesso permitido
      
    } else if (command == "Acesso_negado") {
      for (int i = 0; i < 3; i++){
        digitalWrite(ledPin_off, HIGH);  // Apaga o LED acesso negado
        delay(50); // Mantem o LED de acesso negado ligado por 0.05 segundo.
        digitalWrite(ledPin_off, LOW);  // Apaga o LED acesso negado
        delay(50);

        digitalWrite(buzzerPin, HIGH); //Ligar Buzzer
        delay(50); // Liga o buzzer ligado por 0.05 segundos.
        digitalWrite(buzzerPin, LOW); //Desliga Buzzer
        delay(50); // Desliga o buzzer (0.05 segundos)
      }
    }
  }
}
