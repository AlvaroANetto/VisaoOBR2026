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
        processarComando(buffer);
      }
      bufferIdx = 0;  
    }
    else if (c != '\r' && bufferIdx < 63) {
      buffer[bufferIdx++] = c;
    }
  }
}

void processarComando(char* cmd) {
  // 1. Mostra exatamente o que chegou (com colchetes para ver espaços ocultos)
  Serial.print("📦 Bruto recebido: ["); 
  Serial.print(cmd); 
  Serial.println("]");

  // 2. Verifica se começa com 'A'
  if (cmd[0] != 'A') {
    Serial.println("❌ ERRO 1: O comando não começa com a letra 'A'");
    return;
  }

  // 3. Procura o asterisco '*' que separa o valor do checksum
  char* asterisk = strchr(cmd, '*');
  if (!asterisk) {
    Serial.println("❌ ERRO 2: Não encontrei o caractere '*' (checksum ausente)");
    return;
  }

  // 4. Separa a string e calcula o checksum
  *asterisk = '\0'; // Transforma o '*' em fim de string para o atof funcionar
  int checksumRecebido = strtol(asterisk + 1, NULL, 16);

  int checksumCalculado = 0;
  for (char* p = cmd; *p; p++) {
    checksumCalculado += (unsigned char)*p; // Cast para evitar valores negativos
  }
  checksumCalculado &= 0xFF; // Garante que fique em 8 bits (0-255)

  // 5. Valida o checksum
  if (checksumRecebido != checksumCalculado) {
    Serial.print("❌ ERRO 3: Checksum inválido! | Recebido: 0x");
    Serial.print(checksumRecebido, HEX);
    Serial.print(" | Calculado: 0x");
    Serial.println(checksumCalculado, HEX);
    return; // <-- Provavelmente está parando aqui!
  }

  // 6. SE CHEGOU AQUI, O PACOTE É 100% VÁLIDO!
  float anguloRecebido = atof(cmd + 1); // Pula o 'A' e converte o resto para float
  
  Serial.print("✅ SUCESSO! Ângulo cru: "); 
  Serial.println(anguloRecebido, 4);

  // --- A PARTIR DAQUI É O SEU CONTROLE NORMAL ---
  
  // Deadband
  if (fabs(anguloRecebido) < 0.08) { // 0.08 é o DEADBAND
    anguloRecebido = 0.0;
  }

  // Filtro Exponencial
  if (!filtroInicializado) {
    anguloFiltrado = anguloRecebido;
    filtroInicializado = true;
  } else {
    anguloFiltrado = 0.75 * anguloFiltrado + 0.25 * anguloRecebido;
  }

  Serial.print("🎯 Ângulo Filtrado: "); 
  Serial.println(anguloFiltrado, 4);

  // Envia para os motores
  controlarMotores(anguloFiltrado);
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
