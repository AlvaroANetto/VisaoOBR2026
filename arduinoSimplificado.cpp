// =========================
// CONFIGURAÇÃO BÁSICA
// =========================
#define VELOCIDADE_BASE 150 // Aumentado para vencer o atrito estático
#define KP 120.0            // Ajuste fino (Proporcional)
#define KD 180.0            // Ajuste fino (Derivativo - Antecipa curvas)

float erro_anterior = 0;
unsigned long tempo_anterior = 0;

char buffer[64];
int bufferIdx = 0;

void setup() {
  Serial.begin(115200);    
  Serial1.begin(115200);   
  Serial.println("Arduino pronto. Aguardando dados...");
  tempo_anterior = micros();
}

void loop() {
  receberSerial();
}

void receberSerial() {
  while (Serial1.available()) {
    char c = Serial1.read();
    
    if (c == '\n') {
      buffer[bufferIdx] = '\0';  
      if (bufferIdx > 0) {
        processar(buffer);
      }
      bufferIdx = 0;  
    }
    else if (c != '\r' && bufferIdx < 63) {
      buffer[bufferIdx++] = c;
    }
  }
}

void processar(char* msg) {
  // 1. Separar Payload do Checksum
  char* asterisco = strchr(msg, '*');
  if (!asterisco) return; // Pacote malformado
  
  *asterisco = '\0'; 
  char* payload = msg;
  char* checksum_recv_str = asterisco + 1;
  
  // 2. Calcular Checksum local
  int calc_sum = 0;
  for (int i = 0; payload[i] != '\0'; i++) {
    calc_sum += payload[i];
  }
  calc_sum &= 0xFF;
  
  // 3. Validar Checksum
  int recv_sum = (int)strtol(checksum_recv_str, NULL, 16);
  if (calc_sum != recv_sum) {
    // Serial.println("Erro de Checksum! Pacote ignorado.");
    return; 
  }
  
  // 4. Extrair Ângulo
  float angulo = 0;
  if (payload[0] == 'A') {
    angulo = atof(payload + 1);
  }
  
  controlarMotores(angulo);
}

void controlarMotores(float erro) {
  // Cálculo do Tempo (dt) para a Derivada
  unsigned long tempo_atual = micros();
  float dt = (tempo_atual - tempo_anterior) / 1000000.0;
  tempo_anterior = tempo_atual;
  if (dt <= 0) dt = 0.02; 
  
  // Controle PD
  float derivada = (erro - erro_anterior) / dt;
  erro_anterior = erro;
  
  float correcao = (erro * KP) + (derivada * KD);
  
  int velEsq = VELOCIDADE_BASE + correcao;
  int velDir = VELOCIDADE_BASE - correcao;
  
  velEsq = constrain(velEsq, -255, 255);
  velDir = constrain(velDir, -255, 255);
  
  // Aplica aos motores (Adapte para as funções do seu Driver)
  if (velEsq >= 0) motor_esq(velEsq);
  else motor_esqTras(abs(velEsq));
  
  if (velDir >= 0) motor_dir(velDir);
  else motor_dirTras(abs(velDir));
}
