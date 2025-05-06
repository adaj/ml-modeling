# Quiz: Aula 7 - Desenvolvimento de Modelo e Avaliação Offline (Múltipla Escolha)

Este quiz contém três questões de múltipla escolha de nível intermediário/avançado para cada um dos 8 tópicos abordados na Aula 7. Cada questão possui uma alternativa correta e outras que podem parecer similares. O objetivo é avaliar a compreensão e a capacidade de aplicação dos conceitos apresentados.

---

## Tópico 1: Seleção de Baseline para "Sanity Checking"

1.  **Questão**: Em um projeto de classificação de imagens médicas para detectar uma doença rara, um colega sugere usar um modelo de "Random Predictor" como único baseline. Qual das seguintes afirmações melhor descreve a inadequação dessa escolha e propõe alternativas mais significativas?
    a)  O Random Predictor é adequado, pois estabelece um limite inferior absoluto de desempenho, e qualquer modelo acima dele é um ganho. Alternativas como um modelo que sempre prevê a classe majoritária seriam muito complexas para um baseline.
    b)  O Random Predictor é inadequado porque, para uma doença rara, ele terá uma acurácia artificialmente alta ao prever majoritariamente a ausência da doença. Baselines mais significativos seriam: (1) um modelo de Regressão Logística simples com features básicas extraídas das imagens (ex: histograma de intensidade) e (2) um modelo que sempre prevê a classe mais frequente (não-doente), para entender o quão melhor o modelo precisa ser em relação à simples prevalência.
    c)  O Random Predictor é inadequado, mas a melhor alternativa seria um modelo de Deep Learning pré-treinado (como ResNet) usado diretamente como baseline, pois representa o estado da arte.
    d)  O Random Predictor é uma boa escolha inicial, mas deveria ser complementado por um baseline que prevê sempre a classe minoritária (doente), para focar na capacidade de detecção da doença rara.

2.  **Questão**: Você desenvolve um modelo de regressão (rede neural profunda) para prever o valor de imóveis, alcançando um R² de 0.65 na validação. Um baseline de Regressão Linear com 5 features atinge R² de 0.60. Qual interpretação e próximo passo são mais prudentes?
    a)  O modelo complexo é claramente superior, pois 0.05 de R² é um ganho significativo. Deve-se prosseguir com o deploy do modelo complexo imediatamente.
    b)  A diferença de 0.05 no R² é pequena, sugerindo que a complexidade adicional da rede neural pode não estar trazendo um benefício substancial ou que há problemas no modelo complexo. Próximos passos: investigar se o modelo complexo está sofrendo overfitting, analisar os resíduos de ambos os modelos, e verificar se as features usadas pelo baseline são realmente as mais informativas para o modelo complexo.
    c)  O baseline de Regressão Linear é surpreendentemente bom, indicando que o problema é mais simples do que o esperado. Deve-se abandonar o modelo complexo e focar em otimizar o baseline.
    d)  Ambos os modelos são ruins, pois um R² de 0.65 é baixo para previsão de imóveis. É necessário coletar mais dados antes de qualquer outra ação.

3.  **Questão**: O conceito de "valor incremental" na seleção de baselines refere-se principalmente a:
    a)  A diferença absoluta na métrica de desempenho principal entre o modelo proposto e o baseline, independentemente de outros fatores.
    b)  A avaliação de se o ganho de desempenho de um modelo mais complexo sobre um baseline justifica seus custos adicionais (computacionais, de manutenção, de interpretabilidade) e a complexidade introduzida.
    c)  A quantidade de novas features que um modelo complexo consegue utilizar em comparação com um baseline simples.
    d)  O aumento no tamanho do conjunto de dados necessário para treinar o modelo complexo em comparação com o baseline.

---

## Tópico 2: Modelagem de Arquiteturas Neurais (ex com Deep Learning)

1.  **Questão**: Para análise de sentimento em longos documentos de texto, qual arquitetura neural e seus componentes chave seriam mais adequados para capturar dependências de longo alcance e nuances contextuais, em comparação com alternativas menos eficazes?
    a)  Uma Rede Neural Convolucional (CNN) 1D seria ideal, pois suas convoluções locais podem capturar sequências de palavras importantes, e o pooling ajuda a resumir o sentimento geral.
    b)  Uma arquitetura baseada em Transformers, devido aos seus mecanismos de auto-atenção (self-attention) que podem ponderar a importância de todas as palavras no documento, independentemente da distância, superando as limitações de RNNs tradicionais em manter o contexto por longas sequências.
    c)  Uma Rede Neural Recorrente (RNN) simples (não LSTM ou GRU), pois é computacionalmente mais leve e suficiente para capturar a ordem das palavras.
    d)  Um Perceptron Multicamadas (MLP) aplicado a uma representação Bag-of-Words do texto, pois a ordem das palavras não é crucial para o sentimento em documentos longos.

2.  **Questão**: Ao projetar uma CNN para segmentação de imagens com um dataset limitado, qual combinação de estratégias de modelagem e regularização seria mais eficaz para mitigar overfitting e buscar bom desempenho?
    a)  Usar uma arquitetura muito profunda e larga, sem data augmentation, e treinar por muitas épocas para garantir que o modelo aprenda todas as nuances dos poucos dados disponíveis.
    b)  Utilizar uma arquitetura mais simples (menos camadas/filtros), aplicar data augmentation agressiva (rotações, flips, zoom, etc.), e incorporar camadas de Dropout e Batch Normalization para regularizar o treinamento e estabilizar os gradientes.
    c)  Focar apenas em Batch Normalization, pois ela por si só previne overfitting ao normalizar as ativações, tornando Dropout e data augmentation redundantes.
    d)  Aumentar o tamanho do batch ao máximo possível e usar um learning rate muito pequeno, pois isso garante uma convergência suave e evita que o modelo memorize os dados de treino.

3.  **Questão**: O conceito de "hierarquia de features" em CNNs para reconhecimento de objetos implica que:
    a)  Todas as camadas convolucionais aprendem o mesmo tipo de features, mas com diferentes níveis de abstração, sendo as camadas finais responsáveis por refinar as features das camadas iniciais.
    b)  As camadas convolucionais iniciais aprendem features de baixo nível (ex: bordas, texturas), as camadas intermediárias combinam essas features para detectar partes de objetos (ex: olhos, rodas), e as camadas mais profundas (e densas) reconhecem objetos completos com base nessas partes.
    c)  A hierarquia é definida pela ordem em que as classes são aprendidas, com classes mais simples sendo detectadas nas camadas iniciais e classes mais complexas nas camadas finais.
    d)  As camadas densas no final da CNN são as únicas responsáveis por aprender features hierárquicas, enquanto as camadas convolucionais atuam apenas como extratores de features genéricas.

---

## Tópico 3: Treinamento de Modelos

1.  **Questão**: Qual afirmação descreve mais precisamente a interação entre `learning rate` (LR), `batch size` (BS) e otimizadores adaptativos (ex: Adam) no treinamento de redes neurais?
    a)  Um LR alto sempre acelera a convergência, independentemente do BS. Otimizadores adaptativos eliminam a necessidade de ajustar o LR.
    b)  Um BS muito pequeno com LR alto pode causar instabilidade e divergência. Um BS muito grande com LR pequeno pode levar a uma convergência lenta para mínimos locais subótimos. Otimizadores adaptativos ajustam o LR por parâmetro, ajudando a lidar com gradientes esparsos e diferentes escalas, mas ainda se beneficiam de um LR global bem escolhido.
    c)  Otimizadores adaptativos como Adam tornam o `batch size` irrelevante, pois normalizam os gradientes internamente.
    d)  A melhor estratégia é sempre usar o maior `batch size` que a memória permitir e um `learning rate` fixo e pequeno, pois isso garante a convergência mais estável, e otimizadores adaptativos não são necessários.

2.  **Questão**: Qual das seguintes opções descreve corretamente o mecanismo e o impacto principal da regularização L1 em comparação com L2 e Dropout?
    a)  L1 adiciona o quadrado dos pesos à função de perda, levando a pesos menores e mais difusos. L2 adiciona o valor absoluto dos pesos, promovendo esparsidade. Dropout zera aleatoriamente as ativações, o que não afeta os pesos diretamente.
    b)  L1 adiciona o valor absoluto dos pesos à função de perda, o que pode levar alguns pesos a zero (promovendo esparsidade e seleção de features). L2 adiciona o quadrado dos pesos, tendendo a diminuir todos os pesos de forma mais homogênea. Dropout desativa neurônios aleatoriamente durante o treino, forçando a rede a aprender representações mais robustas e redundantes.
    c)  L1 e L2 são idênticas em seu efeito nos pesos, apenas diferem na complexidade computacional. Dropout é uma técnica aplicada apenas na camada de entrada para reduzir o ruído dos dados.
    d)  Dropout ajusta os pesos para serem menores, L1 ajusta as ativações para serem esparsas, e L2 garante que a matriz de pesos seja ortogonal.

3.  **Questão**: Sobre a técnica de `Early Stopping`, qual afirmação é a mais precisa e completa?
    a)  `Early Stopping` interrompe o treinamento assim que a perda de treinamento começa a aumentar, prevenindo qualquer possibilidade de overfitting e garantindo o melhor modelo possível.
    b)  `Early Stopping` monitora uma métrica no conjunto de validação (ex: perda de validação) e interrompe o treinamento quando essa métrica não melhora por um número especificado de épocas (`patience`), retornando o modelo com o melhor desempenho na validação. É uma forma de regularização implícita, mas pode interromper o treinamento prematuramente se a `patience` for muito baixa ou se a métrica de validação for muito ruidosa.
    c)  `Early Stopping` é configurado apenas com o parâmetro `monitor` (ex: `val_loss`) e sempre interrompe o treinamento na primeira vez que a métrica monitorada piora, sendo a forma mais agressiva de evitar overfitting.
    d)  `Early Stopping` só é útil quando a perda de treinamento e a perda de validação divergem significativamente; se ambas estiverem diminuindo, não há necessidade de usá-lo.

---

## Tópico 4: Debugging Através do Monitoramento de Experimentos

1.  **Questão**: Durante o treinamento de um modelo de classificação, a perda de treinamento continua diminuindo, mas a acurácia de validação estagna e depois cai, enquanto a perda de validação aumenta. Qual é a causa mais provável e uma métrica/visualização chave para diagnóstico?
    a)  Causa: Underfitting severo. Diagnóstico: Verificar se a perda de treinamento também está alta e estagnada.
    b)  Causa: Overfitting. O modelo está memorizando os dados de treino e não generaliza. Diagnóstico: Comparar as curvas de aprendizado (perda/acurácia) de treino e validação; a divergência delas é um sinal clássico.
    c)  Causa: Learning rate muito baixo. Diagnóstico: Observar se a perda de treinamento está diminuindo muito lentamente.
    d)  Causa: Problema na implementação da função de perda. Diagnóstico: Inspecionar o código da função de perda e testá-la com exemplos simples.

2.  **Questão**: Além de perda e acurácia, qual dos seguintes é um exemplo de informação crucial a ser logada durante o treinamento de modelos de deep learning para facilitar o debugging de problemas como "dying ReLUs" ou gradientes explosivos/desvanecentes?
    a)  O tempo total de execução de cada época, para otimizar a velocidade do treinamento.
    b)  Histogramas das ativações das camadas e a norma dos gradientes por camada, para identificar se neurônios não estão ativando ou se os gradientes estão se tornando muito pequenos/grandes.
    c)  O número de parâmetros do modelo, para garantir que a complexidade está dentro do esperado.
    d)  A versão do Python e das bibliotecas utilizadas, para garantir a reprodutibilidade do ambiente.

3.  **Questão**: Como sistemas de rastreamento de experimentos (ex: MLflow, W&B) facilitam primariamente a reprodutibilidade e o debugging colaborativo?
    a)  Automatizando completamente o processo de deploy, eliminando a necessidade de intervenção manual.
    b)  Fornecendo um ambiente de desenvolvimento integrado (IDE) específico para machine learning.
    c)  Registrando automaticamente parâmetros, métricas, artefatos e referências ao código para cada execução, permitindo comparar experimentos, identificar o impacto de mudanças e compartilhar resultados e configurações de forma organizada com a equipe.
    d)  Gerando automaticamente relatórios de bugs e sugestões de correção para o código do modelo.

---

## Tópico 5: Apresentação e Exemplificação: MLflow e Weights & Biases

1.  **Questão**: Qual a principal distinção e complementaridade entre os componentes "MLflow Projects" e "MLflow Models" no ciclo de vida de MLOps?
    a.  MLflow Projects define o ambiente de execução e os pontos de entrada para treinar modelos, enquanto MLflow Models define um formato padrão para empacotar os modelos treinados para que possam ser usados em diversas ferramentas de downstream (inferência/deploy). Eles se complementam ao permitir que um Projeto MLflow produza um Modelo MLflow.
    b.  MLflow Projects é usado para registrar modelos no Model Registry, enquanto MLflow Models é usado para rastrear métricas de treinamento.
    c.  MLflow Projects é um formato para empacotar modelos, e MLflow Models é uma UI para visualizar experimentos.
    d. Ambos são formatos para empacotar código de treinamento, mas MLflow Projects é para Python e MLflow Models é para R.

2.  **Questão**: Qual a importância crucial do versionamento e dos estágios (ex: Staging, Production) no "MLflow Model Registry" para o gerenciamento de modelos em um ambiente corporativo?
    a)  Permitem apenas armazenar diferentes modelos, mas não gerenciar seu ciclo de vida ou qualidade.
    b)  Facilitam a organização e o controle dos modelos, permitindo testar novas versões em "Staging" antes de promovê-las para "Production", e reverter para versões anteriores estáveis em caso de problemas, garantindo a governança e a confiabilidade dos modelos em produção.
    c)  São funcionalidades apenas para documentação, sem impacto prático na implantação ou rollback de modelos.
    d)  O versionamento é automático e não requer intervenção, e os estágios são apenas tags informativas sem funcionalidade de controle de fluxo.

3.  **Questão**: Se o foco principal de um projeto de pesquisa em deep learning é a iteração rápida, visualizações ricas e otimização de hiperparâmetros colaborativa, qual ferramenta tenderia a ser preferida e por quê, em contraste com um cenário de padronização do ciclo de vida de ML em uma grande organização com necessidade de auto-hospedagem?
    a)  MLflow seria preferido para pesquisa rápida devido à sua UI simples, enquanto W&B seria para a organização devido à sua natureza open-source.
    b)  Weights & Biases (W&B) tenderia a ser preferido para pesquisa rápida devido à sua UI polida, dashboards automáticos e sweeps integrados. MLflow seria mais inclinado para a organização devido à sua modularidade, capacidade de auto-hospedagem e foco no ciclo de vida completo, incluindo registro para deploy.
    c)  Ambas as ferramentas são idênticas em funcionalidades e preferências, a escolha é puramente baseada no custo.
    d)  MLflow é melhor para visualizações ricas, e W&B é melhor para auto-hospedagem e CI/CD.

---

## Tópico 6: Seleção de Hiperparâmetros e AutoML

1.  **Questão**: Qual das seguintes afirmações descreve mais precisamente por que a Otimização Bayesiana pode ser mais eficiente que Grid Search ou Random Search para seleção de hiperparâmetros?
    a)  A Otimização Bayesiana experimenta todas as combinações possíveis, garantindo o ótimo global, mas de forma mais rápida que o Grid Search.
    b)  A Otimização Bayesiana constrói um modelo probabilístico da função objetivo (desempenho vs. hiperparâmetros) e usa esse modelo para escolher os próximos pontos a serem avaliados de forma mais inteligente, focando em regiões promissoras, o que geralmente requer menos avaliações do modelo real.
    c)  A Otimização Bayesiana seleciona hiperparâmetros de forma puramente aleatória, mas utiliza um número muito maior de iterações que o Random Search, garantindo melhor cobertura.
    d)  A Otimização Bayesiana só funciona para modelos Bayesianos e não é aplicável a redes neurais ou árvores de decisão.

2.  **Questão**: Neural Architecture Search (NAS), um componente de algumas ferramentas de AutoML, refere-se principalmente a:
    a)  Um método para buscar automaticamente os melhores hiperparâmetros de uma arquitetura de rede neural já definida.
    b)  O processo de automatizar o design da própria arquitetura de uma rede neural, como o número e tipo de camadas, conexões, etc.
    c)  Uma técnica para selecionar automaticamente o melhor algoritmo de otimização (ex: Adam, SGD) para uma dada arquitetura.
    d)  Um sistema para buscar automaticamente o melhor conjunto de dados de treinamento para uma arquitetura específica.

3.  **Questão**: Mesmo com o avanço do AutoML, por que a expertise humana continua crucial em um projeto de MLOps?
    a)  Porque o AutoML ainda não consegue realizar a divisão dos dados em treino, validação e teste automaticamente.
    b)  Porque a definição clara do problema de negócio, a curadoria e engenharia de features relevantes, a interpretação dos resultados do modelo AutoML e a consideração de aspectos éticos e de fairness são tarefas que exigem julgamento e conhecimento de domínio humano.
    c)  Porque as ferramentas de AutoML geralmente produzem código de baixa qualidade que precisa ser reescrito por especialistas.
    d)  Porque o AutoML só funciona para problemas de classificação e não para regressão ou outros tipos de tarefas de ML.

---

## Tópico 7: Avaliação Offline (confusion matrix, learning curves, cross-validation)

1.  **Questão**: Em um problema de detecção de fraude (classificação binária altamente desbalanceada), por que a acurácia é uma métrica inadequada e qual combinação de métricas seria mais informativa?
    a)  Acurácia é inadequada porque será artificialmente alta se o modelo simplesmente prever a classe majoritária (não-fraude). Métricas como Precisão, Recall (para a classe fraude) e F1-Score são mais informativas, pois focam na capacidade do modelo de identificar corretamente os casos raros de fraude, considerando os custos de falsos positivos e falsos negativos.
    b)  Acurácia é a melhor métrica, pois reflete o desempenho geral. Outras métricas como Precisão e Recall são muito específicas e podem confundir a avaliação.
    c)  Acurácia é inadequada, mas a única alternativa viável é o ROC AUC, pois ele resume o desempenho em todos os limiares.
    d)  Acurácia é adequada se o dataset for rebalanceado usando técnicas como SMOTE antes da avaliação.

2.  **Questão**: Ao analisar curvas de aprendizado, você observa que a perda de treinamento está baixa e continua diminuindo, enquanto a perda de validação, após diminuir inicialmente, começa a aumentar. O que isso indica e qual uma estratégia de mitigação apropriada?
    a)  Indica underfitting. Estratégia: Aumentar a complexidade do modelo ou treinar por mais épocas.
    b)  Indica overfitting. Estratégia: Aumentar a regularização (ex: L2, Dropout), obter mais dados de treinamento, ou reduzir a complexidade do modelo.
    c)  Indica que o learning rate está muito alto. Estratégia: Reduzir o learning rate e reiniciar o treinamento.
    d)  Indica um bom ajuste do modelo. Nenhuma ação é necessária, pois a perda de treinamento está baixa.

3.  **Questão**: Qual o principal benefício da Validação Cruzada K-Fold em comparação com uma única divisão treino-validação, e quando a Validação Cruzada Estratificada (Stratified K-Fold) é preferível?
    a)  K-Fold é mais rápida de executar. Stratified K-Fold é usada apenas para problemas de regressão.
    b)  K-Fold fornece uma estimativa mais robusta (menor variância) do desempenho de generalização do modelo, utilizando os dados de forma mais eficiente. Stratified K-Fold é preferível quando o dataset é desbalanceado, pois garante que cada fold mantenha aproximadamente a mesma proporção de amostras de cada classe que o conjunto original.
    c)  K-Fold sempre resulta em melhor desempenho do modelo. Stratified K-Fold é mais complexa e raramente oferece vantagens.
    d)  K-Fold é usada para otimização de hiperparâmetros e uma única divisão treino-validação para a avaliação final. Stratified K-Fold não é uma técnica padrão.

---

## Tópico 8: Calibração do Modelo Final

1.  **Questão**: Por que um modelo com alta acurácia em classificação pode ter probabilidades mal calibradas, e em qual aplicação probabilidades bem calibradas seriam mais críticas?
    a)  Probabilidades mal calibradas ocorrem apenas em modelos com baixa acurácia. Em aplicações de marketing, a acurácia é sempre mais importante que a calibração.
    b)  Muitos algoritmos (ex: redes neurais, SVMs) otimizam para acurácia de classificação, mas suas saídas brutas (scores) não são necessariamente probabilidades verdadeiras. Probabilidades bem calibradas são críticas em diagnóstico médico, onde a confiança na previsão (ex: 80% de chance de doença) influencia diretamente decisões de tratamento com consequências significativas.
    c)  Se um modelo tem alta acurácia, suas probabilidades são inerentemente bem calibradas. A calibração é um passo opcional para melhorar a interpretabilidade, mas não afeta a tomada de decisão.
    d)  Probabilidades mal calibradas são um sinal de overfitting severo, e o modelo deve ser descartado. Em qualquer aplicação, a acurácia é o único fator determinante.

2.  **Questão**: Ao interpretar um Diagrama de Confiabilidade (Reliability Diagram), como você identificaria um modelo excessivamente confiante?
    a)  A curva de calibração do modelo estaria consistentemente acima da diagonal de calibração perfeita.
    b)  A curva de calibração do modelo estaria consistentemente abaixo da diagonal de calibração perfeita (ex: para previsões com probabilidade média de 0.8, a fração real de positivos é apenas 0.6).
    c)  A curva de calibração do modelo seguiria perfeitamente a diagonal.
    d)  O diagrama mostraria uma dispersão aleatória de pontos sem seguir um padrão claro em relação à diagonal.

3.  **Questão**: Comparando Platt Scaling e Isotonic Regression para calibração de modelo, qual afirmação é mais precisa?
    a)  Platt Scaling é não-paramétrico e mais flexível, enquanto Isotonic Regression ajusta uma regressão logística, sendo mais restritiva.
    b)  Platt Scaling ajusta uma regressão logística aos outputs do modelo (adequado para saídas sigmoidais), enquanto Isotonic Regression ajusta uma função não-paramétrica não-decrescente, sendo mais flexível e potencialmente mais poderosa, mas pode exigir mais dados para calibração para evitar overfitting.
    c)  Ambas as técnicas são idênticas em funcionamento e resultados, sendo intercambiáveis em todos os cenários.
    d)  Isotonic Regression é usada apenas para modelos de regressão, e Platt Scaling apenas para classificadores SVM.

---

## Gabarito

(Será preenchido na próxima etapa)

---




## Gabarito

### Tópico 1: Seleção de Baseline para "Sanity Checking"
1.  **Resposta Correta**: b) O Random Predictor é inadequado porque, para uma doença rara, ele terá uma acurácia artificialmente alta ao prever majoritariamente a ausência da doença. Baselines mais significativos seriam: (1) um modelo de Regressão Logística simples com features básicas extraídas das imagens (ex: histograma de intensidade) e (2) um modelo que sempre prevê a classe mais frequente (não-doente), para entender o quão melhor o modelo precisa ser em relação à simples prevalência.
2.  **Resposta Correta**: b) A diferença de 0.05 no R² é pequena, sugerindo que a complexidade adicional da rede neural pode não estar trazendo um benefício substancial ou que há problemas no modelo complexo. Próximos passos: investigar se o modelo complexo está sofrendo overfitting, analisar os resíduos de ambos os modelos, e verificar se as features usadas pelo baseline são realmente as mais informativas para o modelo complexo.
3.  **Resposta Correta**: b) A avaliação de se o ganho de desempenho de um modelo mais complexo sobre um baseline justifica seus custos adicionais (computacionais, de manutenção, de interpretabilidade) e a complexidade introduzida.

### Tópico 2: Modelagem de Arquiteturas Neurais (ex com Deep Learning)
1.  **Resposta Correta**: b) Uma arquitetura baseada em Transformers, devido aos seus mecanismos de auto-atenção (self-attention) que podem ponderar a importância de todas as palavras no documento, independentemente da distância, superando as limitações de RNNs tradicionais em manter o contexto por longas sequências.
2.  **Resposta Correta**: b) Utilizar uma arquitetura mais simples (menos camadas/filtros), aplicar data augmentation agressiva (rotações, flips, zoom, etc.), e incorporar camadas de Dropout e Batch Normalization para regularizar o treinamento e estabilizar os gradientes.
3.  **Resposta Correta**: b) As camadas convolucionais iniciais aprendem features de baixo nível (ex: bordas, texturas), as camadas intermediárias combinam essas features para detectar partes de objetos (ex: olhos, rodas), e as camadas mais profundas (e densas) reconhecem objetos completos com base nessas partes.

### Tópico 3: Treinamento de Modelos
1.  **Resposta Correta**: b) Um BS muito pequeno com LR alto pode causar instabilidade e divergência. Um BS muito grande com LR pequeno pode levar a uma convergência lenta para mínimos locais subótimos. Otimizadores adaptativos ajustam o LR por parâmetro, ajudando a lidar com gradientes esparsos e diferentes escalas, mas ainda se beneficiam de um LR global bem escolhido.
2.  **Resposta Correta**: b) L1 adiciona o valor absoluto dos pesos à função de perda, o que pode levar alguns pesos a zero (promovendo esparsidade e seleção de features). L2 adiciona o quadrado dos pesos, tendendo a diminuir todos os pesos de forma mais homogênea. Dropout desativa neurônios aleatoriamente durante o treino, forçando a rede a aprender representações mais robustas e redundantes.
3.  **Resposta Correta**: b) `Early Stopping` monitora uma métrica no conjunto de validação (ex: perda de validação) e interrompe o treinamento quando essa métrica não melhora por um número especificado de épocas (`patience`), retornando o modelo com o melhor desempenho na validação. É uma forma de regularização implícita, mas pode interromper o treinamento prematuramente se a `patience` for muito baixa ou se a métrica de validação for muito ruidosa.

### Tópico 4: Debugging Através do Monitoramento de Experimentos
1.  **Resposta Correta**: b) Causa: Overfitting. O modelo está memorizando os dados de treino e não generaliza. Diagnóstico: Comparar as curvas de aprendizado (perda/acurácia) de treino e validação; a divergência delas é um sinal clássico.
2.  **Resposta Correta**: b) Histogramas das ativações das camadas e a norma dos gradientes por camada, para identificar se neurônios não estão ativando ou se os gradientes estão se tornando muito pequenos/grandes.
3.  **Resposta Correta**: c) Registrando automaticamente parâmetros, métricas, artefatos e referências ao código para cada execução, permitindo comparar experimentos, identificar o impacto de mudanças e compartilhar resultados e configurações de forma organizada com a equipe.

### Tópico 5: Apresentação e Exemplificação: MLflow e Weights & Biases
1.  **Resposta Correta**: a) MLflow Projects define o ambiente de execução e os pontos de entrada para treinar modelos, enquanto MLflow Models define um formato padrão para empacotar os modelos treinados para que possam ser usados em diversas ferramentas de downstream (inferência/deploy). Eles se complementam ao permitir que um Projeto MLflow produza um Modelo MLflow.
2.  **Resposta Correta**: b) Facilitam a organização e o controle dos modelos, permitindo testar novas versões em "Staging" antes de promovê-las para "Production", e reverter para versões anteriores estáveis em caso de problemas, garantindo a governança e a confiabilidade dos modelos em produção.
3.  **Resposta Correta**: b) Weights & Biases (W&B) tenderia a ser preferido para pesquisa rápida devido à sua UI polida, dashboards automáticos e sweeps integrados. MLflow seria mais inclinado para a organização devido à sua modularidade, capacidade de auto-hospedagem e foco no ciclo de vida completo, incluindo registro para deploy.

### Tópico 6: Seleção de Hiperparâmetros e AutoML
1.  **Resposta Correta**: b) A Otimização Bayesiana constrói um modelo probabilístico da função objetivo (desempenho vs. hiperparâmetros) e usa esse modelo para escolher os próximos pontos a serem avaliados de forma mais inteligente, focando em regiões promissoras, o que geralmente requer menos avaliações do modelo real.
2.  **Resposta Correta**: b) O processo de automatizar o design da própria arquitetura de uma rede neural, como o número e tipo de camadas, conexões, etc.
3.  **Resposta Correta**: b) Porque a definição clara do problema de negócio, a curadoria e engenharia de features relevantes, a interpretação dos resultados do modelo AutoML e a consideração de aspectos éticos e de fairness são tarefas que exigem julgamento e conhecimento de domínio humano.

### Tópico 7: Avaliação Offline (confusion matrix, learning curves, cross-validation)
1.  **Resposta Correta**: a) Acurácia é inadequada porque será artificialmente alta se o modelo simplesmente prever a classe majoritária (não-fraude). Métricas como Precisão, Recall (para a classe fraude) e F1-Score são mais informativas, pois focam na capacidade do modelo de identificar corretamente os casos raros de fraude, considerando os custos de falsos positivos e falsos negativos.
2.  **Resposta Correta**: b) Indica overfitting. Estratégia: Aumentar a regularização (ex: L2, Dropout), obter mais dados de treinamento, ou reduzir a complexidade do modelo.
3.  **Resposta Correta**: b) K-Fold fornece uma estimativa mais robusta (menor variância) do desempenho de generalização do modelo, utilizando os dados de forma mais eficiente. Stratified K-Fold é preferível quando o dataset é desbalanceado, pois garante que cada fold mantenha aproximadamente a mesma proporção de amostras de cada classe que o conjunto original.

### Tópico 8: Calibração do Modelo Final
1.  **Resposta Correta**: b) Muitos algoritmos (ex: redes neurais, SVMs) otimizam para acurácia de classificação, mas suas saídas brutas (scores) não são necessariamente probabilidades verdadeiras. Probabilidades bem calibradas são críticas em diagnóstico médico, onde a confiança na previsão (ex: 80% de chance de doença) influencia diretamente decisões de tratamento com consequências significativas.
2.  **Resposta Correta**: b) A curva de calibração do modelo estaria consistentemente abaixo da diagonal de calibração perfeita (ex: para previsões com probabilidade média de 0.8, a fração real de positivos é apenas 0.6).
3.  **Resposta Correta**: b) Platt Scaling ajusta uma regressão logística aos outputs do modelo (adequado para saídas sigmoidais), enquanto Isotonic Regression ajusta uma função não-paramétrica não-decrescente, sendo mais flexível e potencialmente mais poderosa, mas pode exigir mais dados para calibração para evitar overfitting.

---

