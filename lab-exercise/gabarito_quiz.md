
# Gabarito do Quiz "Desenvolvimento e Avaliação Offline de Modelos"

### Tópico 1: Seleção de Baseline para "Sanity Checking"
1.  **Resposta Correta**:  b) O Random Predictor é inadequado porque, para uma doença rara, ele terá uma acurácia artificialmente alta ao prever majoritariamente a ausência da doença. Baselines mais significativos seriam: (1) um modelo de Regressão Logística simples com features básicas extraídas das imagens (ex: histograma de intensidade) e (2) um modelo que sempre prevê a classe mais frequente (não-doente), para entender o quão melhor o modelo precisa ser em relação à simples prevalência.
2.  **Resposta Correta**:  b) A diferença de 0.05 no R² é pequena, sugerindo que a complexidade adicional da rede neural pode não estar trazendo um benefício substancial ou que há problemas no modelo complexo. Próximos passos: investigar se o modelo complexo está sofrendo overfitting, analisar os resíduos de ambos os modelos, e verificar se as features usadas pelo baseline são realmente as mais informativas para o modelo complexo.
3.  **Resposta Correta**:  b) A avaliação de se o ganho de desempenho de um modelo mais complexo sobre um baseline justifica seus custos adicionais (computacionais, de manutenção, de interpretabilidade) e a complexidade introduzida.

### Tópico 2: Modelagem de Arquiteturas Neurais (ex com Deep Learning)
1.  **Resposta Correta**:  b) Uma arquitetura baseada em Transformers, devido aos seus mecanismos de auto-atenção (self-attention) que podem ponderar a importância de todas as palavras no documento, independentemente da distância, superando as limitações de RNNs tradicionais em manter o contexto por longas sequências.
2.  **Resposta Correta**:  b) Utilizar uma arquitetura mais simples (menos camadas/filtros), aplicar data augmentation agressiva (rotações, flips, zoom, etc.), e incorporar camadas de Dropout e Batch Normalization para regularizar o treinamento e estabilizar os gradientes.
3.  **Resposta Correta**:  b) As camadas convolucionais iniciais aprendem features de baixo nível (ex: bordas, texturas), as camadas intermediárias combinam essas features para detectar partes de objetos (ex: olhos, rodas), e as camadas mais profundas (e densas) reconhecem objetos completos com base nessas partes.

### Tópico 3: Treinamento de Modelos
1.  **Resposta Correta**:  b) Um BS muito pequeno com LR alto pode causar instabilidade e divergência. Um BS muito grande com LR pequeno pode levar a uma convergência lenta para mínimos locais subótimos. Otimizadores adaptativos ajustam o LR por parâmetro, ajudando a lidar com gradientes esparsos e diferentes escalas, mas ainda se beneficiam de um LR global bem escolhido.
2.  **Resposta Correta**:  b) L1 adiciona o valor absoluto dos pesos à função de perda, o que pode levar alguns pesos a zero (promovendo esparsidade e seleção de features). L2 adiciona o quadrado dos pesos, tendendo a diminuir todos os pesos de forma mais homogênea. Dropout desativa neurônios aleatoriamente durante o treino, forçando a rede a aprender representações mais robustas e redundantes.
3.  **Resposta Correta**:  b) `Early Stopping` monitora uma métrica no conjunto de validação (ex: perda de validação) e interrompe o treinamento quando essa métrica não melhora por um número especificado de épocas (`patience`), retornando o modelo com o melhor desempenho na validação. É uma forma de regularização implícita, mas pode interromper o treinamento prematuramente se a `patience` for muito baixa ou se a métrica de validação for muito ruidosa.

### Tópico 4: Debugging Através do Monitoramento de Experimentos
1.  **Resposta Correta**:  b) Causa: Overfitting. O modelo está memorizando os dados de treino e não generaliza. Diagnóstico: Comparar as curvas de aprendizado (perda/acurácia) de treino e validação; a divergência delas é um sinal clássico.
2.  **Resposta Correta**:  b) Histogramas das ativações das camadas e a norma dos gradientes por camada, para identificar se neurônios não estão ativando ou se os gradientes estão se tornando muito pequenos/grandes.
3.  **Resposta Correta**:  c) Registrando automaticamente parâmetros, métricas, artefatos e referências ao código para cada execução, permitindo comparar experimentos, identificar o impacto de mudanças e compartilhar resultados e configurações de forma organizada com a equipe.

### Tópico 5: Apresentação e Exemplificação: MLflow e Weights & Biases
1.  **Resposta Correta**:  a) MLflow Projects define o ambiente de execução e os pontos de entrada para treinar modelos, enquanto MLflow Models define um formato padrão para empacotar os modelos treinados para que possam ser usados em diversas ferramentas de downstream (inferência/deploy). Eles se complementam ao permitir que um Projeto MLflow produza um Modelo MLflow.
2.  **Resposta Correta**:  b) Facilitam a organização e o controle dos modelos, permitindo testar novas versões em "Staging" antes de promovê-las para "Production", e reverter para versões anteriores estáveis em caso de problemas, garantindo a governança e a confiabilidade dos modelos em produção.
3.  **Resposta Correta**:  b) Weights & Biases (W&* b) tenderia a ser preferido para pesquisa rápida devido à sua UI polida, dashboards automáticos e sweeps integrados. MLflow seria mais inclinado para a organização devido à sua modularidade, capacidade de auto-hospedagem e foco no ciclo de vida completo, incluindo registro para deploy.

### Tópico 6: Seleção de Hiperparâmetros e AutoML
1.  **Resposta Correta**:  b) A Otimização Bayesiana constrói um modelo probabilístico da função objetivo (desempenho vs. hiperparâmetros) e usa esse modelo para escolher os próximos pontos a serem avaliados de forma mais inteligente, focando em regiões promissoras, o que geralmente requer menos avaliações do modelo real.
2.  **Resposta Correta**:  b) O processo de automatizar o design da própria arquitetura de uma rede neural, como o número e tipo de camadas, conexões, etc.
3.  **Resposta Correta**:  b) Porque a definição clara do problema de negócio, a curadoria e engenharia de features relevantes, a interpretação dos resultados do modelo AutoML e a consideração de aspectos éticos e de fairness são tarefas que exigem julgamento e conhecimento de domínio humano.

### Tópico 7: Avaliação Offline (confusion matrix, learning curves, cross-validation)
1.  **Resposta Correta**:  a) Acurácia é inadequada porque será artificialmente alta se o modelo simplesmente prever a classe majoritária (não-fraude). Métricas como Precisão, Recall (para a classe fraude) e F1-Score são mais informativas, pois focam na capacidade do modelo de identificar corretamente os casos raros de fraude, considerando os custos de falsos positivos e falsos negativos.
2.  **Resposta Correta**:  b) Indica overfitting. Estratégia: Aumentar a regularização (ex: L2, Dropout), obter mais dados de treinamento, ou reduzir a complexidade do modelo.
3.  **Resposta Correta**:  b) K-Fold fornece uma estimativa mais robusta (menor variânci* a) do desempenho de generalização do modelo, utilizando os dados de forma mais eficiente. Stratified K-Fold é preferível quando o dataset é desbalanceado, pois garante que cada fold mantenha aproximadamente a mesma proporção de amostras de cada classe que o conjunto original.

### Tópico 8: Calibração do Modelo Final
1.  **Resposta Correta**:  b) Muitos algoritmos (ex: redes neurais, SVMs) otimizam para acurácia de classificação, mas suas saídas brutas (scores) não são necessariamente probabilidades verdadeiras. Probabilidades bem calibradas são críticas em diagnóstico médico, onde a confiança na previsão (ex: 80% de chance de doenç* a) influencia diretamente decisões de tratamento com consequências significativas.
2.  **Resposta Correta**:  b) A curva de calibração do modelo estaria consistentemente abaixo da diagonal de calibração perfeita (ex: para previsões com probabilidade média de 0.8, a fração real de positivos é apenas 0.6).
3.  **Resposta Correta**:  b) Platt Scaling ajusta uma regressão logística aos outputs do modelo (adequado para saídas sigmoidais), enquanto Isotonic Regression ajusta uma função não-paramétrica não-decrescente, sendo mais flexível e potencialmente mais poderosa, mas pode exigir mais dados para calibração para evitar overfitting.

---
