# Informe técnico — Evaluación Parcial N°2

## Caso A: Predicción de abandono de clientes (Telco Customer Churn)

**Asignatura:** MLY1101 — Machine Learning

**Integrantes:** Benjamin Luengo, Eduardo Rodriguez y Diego Iturra

**Docente:** Jazna Meza

**Formato:** Informe técnico en Markdown

---

## 1. Descripción del problema de negocio

Una empresa de telecomunicaciones necesita comprender qué características presentan los clientes que abandonan el servicio y contar con herramientas que permitan anticipar ese comportamiento.

El problema de negocio consiste en identificar clientes con riesgo de abandono (`Churn`) para apoyar futuras estrategias de retención y, al mismo tiempo, descubrir segmentos de clientes con características similares que permitan orientar acciones más específicas.

La variable objetivo del problema supervisado es `Churn`:

- `No`: el cliente permanece en la empresa.
- `Yes`: el cliente abandona la empresa.

El proyecto aborda el problema desde dos perspectivas:

1. **Aprendizaje supervisado:** predecir si un cliente abandonará o no mediante modelos de clasificación.
2. **Aprendizaje no supervisado:** identificar segmentos de clientes mediante clustering y analizar posteriormente su comportamiento de abandono.

---

## 2. Objetivos del proyecto

### 2.1 Objetivo general

Desarrollar una solución de Machine Learning que permita analizar y predecir el abandono de clientes de una empresa de telecomunicaciones, comparando modelos supervisados e identificando segmentos relevantes mediante aprendizaje no supervisado.

### 2.2 Objetivos específicos

- Comprender la estructura y calidad del conjunto de datos.
- Identificar patrones y variables asociadas al abandono mediante EDA.
- Preparar y transformar los datos para su uso en modelos de Machine Learning.
- Implementar y comparar dos modelos de clasificación supervisada.
- Evaluar los modelos con métricas apropiadas para un problema con desbalance de clases.
- Aplicar K-Means para identificar segmentos de clientes con características similares.
- Evaluar la segmentación mediante Inercia y Silhouette Score.
- Visualizar los clústeres mediante PCA.
- Traducir los resultados a conclusiones útiles para el contexto de negocio.

---

## 3. KPIs del proyecto

### Tasa global de abandono

El dataset contiene 7.043 clientes:

- 5.174 no abandonaron.
- 1.869 abandonaron.
- La tasa global de abandono es **26,54%**.

Este indicador funciona como referencia inicial del abandono observado en la empresa, mientras que la línea base de Accuracy del problema de clasificación es aproximadamente 73,46%.

### Recall de la clase positiva

Mide qué proporción de los clientes que realmente abandonan es detectada correctamente por el modelo. Es especialmente importante porque un falso negativo representa a un cliente que realmente abandona, pero que el modelo clasifica como si fuera a permanecer.

### F1-Score

Permite evaluar el equilibrio entre Precision y Recall, lo que resulta útil considerando el desbalance de clases.

### ROC-AUC

Evalúa la capacidad general del modelo para distinguir entre clientes que abandonan y clientes que permanecen.

### Tasa de abandono por clúster

Se utiliza después de construir los clústeres para comparar el comportamiento de abandono entre los segmentos. `Churn` no fue utilizado para formar los grupos.

---

## 4. Fuente de datos

Se utilizó el archivo `Telco_Customer_Churn_Dataset.csv`.

El conjunto de datos contiene:

- **7.043 registros**.
- **21 variables**.

Las variables incluyen características del cliente, antigüedad (`tenure`), servicios contratados, tipo de contrato, método de pago, cargos mensuales, cargos totales y la variable `Churn`.

La variable `customerID` corresponde a un identificador y no fue utilizada como predictor.

---

## 5. Preparación y calidad de los datos

En la revisión inicial se analizaron dimensiones, tipos de datos, valores nulos, duplicados y distribución de la variable objetivo.

No se detectaron filas completamente duplicadas.

### Tratamiento de `TotalCharges`

`TotalCharges` estaba almacenada como tipo `object`. Se identificaron **11 registros con espacios en blanco**, todos correspondientes a clientes con `tenure = 0`.

Se realizaron las siguientes acciones:

1. Conversión de `TotalCharges` a formato numérico.
2. Conversión de los espacios en blanco a valores faltantes.
3. Reemplazo de esos 11 valores por 0, considerando que correspondían a clientes sin antigüedad acumulada.

Después del tratamiento, `TotalCharges` quedó como `float64` y sin valores faltantes.

---

## 6. Análisis exploratorio de datos (EDA)

El EDA mostró diferencias importantes entre los clientes que abandonan y los que permanecen.

### Variables numéricas

Los clientes que abandonan presentan, en promedio:

- menor antigüedad;
- cargos mensuales más altos;
- menores cargos totales acumulados.

Estas diferencias corresponden a asociaciones observadas y no demuestran causalidad.

### Tipo de contrato

| Tipo de contrato | Tasa de abandono |
|---|---:|
| Month-to-month | 42,71% |
| One year | 11,27% |
| Two year | 2,83% |

Los contratos mes a mes presentan la mayor proporción de abandono.

### Servicio de Internet

| Servicio de Internet | Tasa de abandono |
|---|---:|
| Fiber optic | 41,89% |
| DSL | 18,96% |
| No | 7,40% |

### Método de pago

`Electronic check` presenta la tasa de abandono más alta, con aproximadamente **45,29%**.

### Soporte técnico y seguridad online

- Sin `TechSupport`: aproximadamente **41,64%** de abandono.
- Con `TechSupport`: aproximadamente **15,17%**.
- Sin `OnlineSecurity`: aproximadamente **41,77%**.
- Con `OnlineSecurity`: aproximadamente **14,61%**.

### Principales hallazgos

Los clientes con mayores tasas de abandono tienden a presentar algunas de las siguientes características:

- contrato mes a mes;
- fibra óptica;
- pago mediante `Electronic check`;
- ausencia de soporte técnico;
- ausencia de seguridad online;
- menor antigüedad;
- cargos mensuales más elevados.

---

## 7. Preparación para modelamiento supervisado

Para construir los modelos se realizaron las siguientes transformaciones:

1. Eliminación de `customerID` de las variables predictoras.
2. Transformación de `Churn` a formato numérico: `No = 0`, `Yes = 1`.
3. Codificación de variables categóricas mediante `pd.get_dummies()`.
4. Uso de `drop_first=True` para evitar columnas redundantes.
5. División en 80% entrenamiento y 20% prueba (1.409 registros en el conjunto de prueba).
6. Uso de `stratify=y` para conservar la proporción de clases.
7. Aplicación de `StandardScaler` para la Regresión Logística, ajustado solo con los datos de entrenamiento.

---

## 8. Modelos supervisados

Se implementaron dos modelos de clasificación:

- Regresión Logística.
- Árbol de Decisión.

Ambos fueron evaluados con el mismo conjunto de prueba.

### 8.1 Regresión Logística

| Métrica | Resultado |
|---|---:|
| Accuracy | 80,70% |
| Precision | 65,84% |
| Recall | 56,68% |
| F1-Score | 60,92% |
| ROC-AUC | 84,18% |

Matriz de confusión:

- Verdaderos negativos: **925**.
- Falsos positivos: **110**.
- Falsos negativos: **162**.
- Verdaderos positivos: **212**.

De los 374 clientes del conjunto de prueba que realmente abandonaron, el modelo detectó correctamente 212.

### 8.2 Árbol de Decisión optimizado

El segundo modelo supervisado fue un **Árbol de Decisión**, que clasifica a los clientes mediante reglas construidas a partir de sus características. Para seleccionar la profundidad del árbol, se utilizó `GridSearchCV` sobre el conjunto de entrenamiento, con las siguientes configuraciones:

- **Hiperparámetro evaluado:** `max_depth` con valores `[2, 3, 4, 5, 6, 8, 10, None]`.
- **Criterio de selección:** F1-Score (`scoring="f1"`).
- **Validación cruzada:** 5 particiones (`cv=5`).
- **Configuración seleccionada:** `max_depth = 6`.
- **F1-Score promedio de validación cruzada:** **54,93%**.

La búsqueda se ajustó sobre `X_train` e `y_train`. Después de elegir la configuración, el mejor árbol se evaluó sobre el **mismo conjunto de prueba** empleado para la Regresión Logística. Las métricas siguientes corresponden al conjunto de prueba, no a la validación cruzada.

| Métrica | Resultado en prueba |
|---|---:|
| Accuracy | **80,06%** |
| Precision | **66,67%** |
| Recall | **49,73%** |
| F1-Score | **56,97%** |
| ROC-AUC | **82,19%** |

Matriz de confusión del Árbol de Decisión optimizado:

- Verdaderos negativos: **942**.
- Falsos positivos: **93**.
- Falsos negativos: **188**.
- Verdaderos positivos: **186**.

De los **374** clientes del conjunto de prueba que realmente abandonaron, el Árbol de Decisión detectó correctamente **186** y dejó sin detectar **188**. Aunque su Precision es levemente superior a la de la Regresión Logística, su Recall es menor, algo relevante para el objetivo de anticipar abandonos.

### 8.3 Comparación de modelos supervisados

| Modelo | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Regresión Logística | **80,70%** | 65,84% | **56,68%** | **60,92%** | **84,18%** |
| Árbol de Decisión optimizado | 80,06% | **66,67%** | 49,73% | 56,97% | 82,19% |

La **Regresión Logística** presenta el mejor resultado en Accuracy, Recall, F1-Score y ROC-AUC dentro de los dos modelos evaluados. El Árbol de Decisión optimizado obtiene una Precision ligeramente mayor.

Para el problema de negocio, **Recall** tiene especial relevancia: interesa detectar a los clientes que realmente abandonarán. En el mismo conjunto de prueba, la Regresión Logística detectó **212 de 374** abandonos, frente a **186 de 374** del Árbol de Decisión optimizado: **26 clientes adicionales detectados**.

Sin embargo, incluso la Regresión Logística deja sin detectar **162 de 374** abandonos reales (aproximadamente **43,32%**). Por tanto, su resultado es un apoyo para priorizar análisis y posibles acciones de retención, no una garantía ni un criterio único para decidir sobre clientes individuales.

---

## 9. Aprendizaje no supervisado

Se utilizó **K-Means** para identificar grupos de clientes con características similares.

### Variables utilizadas

- `tenure`.
- `MonthlyCharges`.
- `TotalCharges`.

Las variables fueron escaladas con `StandardScaler` porque K-Means trabaja con distancias y las magnitudes originales son diferentes.

`Churn` no se utilizó para formar los clústeres.

---

## 10. Selección del número de clústeres

Se probaron valores de K entre 2 y 10 utilizando Inercia y Silhouette Score.

| K | Inercia | Silhouette |
|---:|---:|---:|
| 2 | 9712,11 | 0,4797 |
| 3 | 6186,19 | 0,4515 |
| 4 | 4148,48 | 0,4722 |
| 5 | 3111,40 | 0,4437 |

El mayor Silhouette Score se obtuvo con K=2. Sin embargo, K=4 obtuvo un resultado muy cercano y permite una segmentación más detallada. Considerando conjuntamente el comportamiento de la inercia y Silhouette Score, se seleccionó **K = 4**.

---

## 11. Perfil de los clústeres

| Clúster | Cantidad de clientes | tenure promedio | MonthlyCharges promedio | TotalCharges promedio |
|---:|---:|---:|---:|---:|
| 0 | 1.703 | 10,21 | 31,78 | 302,15 |
| 1 | 1.904 | 59,53 | 93,31 | 5.548,65 |
| 2 | 2.276 | 15,42 | 80,78 | 1.251,17 |
| 3 | 1.160 | 53,57 | 34,91 | 1.835,62 |

Interpretación:

- **Clúster 0:** clientes recientes con bajo gasto mensual.
- **Clúster 1:** clientes antiguos con alto gasto.
- **Clúster 2:** clientes recientes con alto gasto mensual.
- **Clúster 3:** clientes antiguos con bajo gasto mensual.

---

## 12. Abandono según clúster

Después de construir los grupos se analizó `Churn` únicamente para interpretar el comportamiento de cada segmento.

| Clúster | No abandona | Abandona |
|---:|---:|---:|
| 0 | 75,34% | 24,66% |
| 1 | 84,61% | 15,39% |
| 2 | 51,76% | 48,24% |
| 3 | 95,00% | 5,00% |

El **Clúster 2** presenta la mayor tasa de abandono:

- 2.276 clientes.
- `tenure` promedio: 15,42 meses.
- `MonthlyCharges` promedio: 80,78.
- Tasa de abandono: **48,24%**.
- 1.098 clientes con `Churn = Yes`.

Este segmento representa un grupo prioritario para profundizar futuras estrategias de retención.

---

## 13. PCA y visualización de clústeres

PCA se utilizó únicamente para visualizar los clústeres en dos dimensiones.

- PC1: **72,69%** de varianza explicada.
- PC2: **25,33%**.
- Varianza explicada acumulada: **98,01%**.

Los dos componentes conservan prácticamente toda la variabilidad de las tres variables utilizadas en el clustering.

La visualización muestra regiones diferenciadas entre los cuatro grupos, aunque existen zonas de frontera con cierto solapamiento.

PCA no creó los clústeres; los grupos fueron construidos previamente mediante K-Means.

---

## 14. Metodología CRISP-DM

### 14.1 Comprensión del negocio

Se definió como problema principal la necesidad de identificar clientes con riesgo de abandonar la empresa para apoyar estrategias de retención.

### 14.2 Comprensión de los datos

Se revisaron dimensiones, tipos de variables, valores faltantes, duplicados, distribución de `Churn` y patrones relevantes mediante EDA.

### 14.3 Preparación de los datos

Se corrigió `TotalCharges`, se eliminaron identificadores del modelamiento, se codificaron variables categóricas, se dividieron los datos y se realizó escalamiento cuando correspondía.

### 14.4 Modelamiento

Se implementaron:

- Regresión Logística.
- Árbol de Decisión.
- K-Means.

### 14.5 Evaluación

Para clasificación se utilizaron Accuracy, Precision, Recall, F1-Score, ROC-AUC y matrices de confusión.

Para K-Means se utilizaron Inercia, Método del Codo y Silhouette Score. PCA se empleó para visualización.

### 14.6 Aplicación al contexto de negocio

Se seleccionó la Regresión Logística como el mejor modelo supervisado entre los evaluados (comparado con el Árbol de Decisión optimizado mediante GridSearchCV) y se identificó al Clúster 2 como el segmento con mayor tasa de abandono.

---

## 15. Consideraciones éticas

Los resultados deben utilizarse de forma responsable.

- Las asociaciones observadas no demuestran causalidad.
- Una predicción de abandono no representa una certeza.
- Pertenecer a un clúster no implica que todos sus integrantes tengan el mismo comportamiento.
- Los modelos no deben utilizarse como único criterio para tomar decisiones que afecten a clientes individuales.
- Deben evitarse decisiones discriminatorias basadas exclusivamente en características personales.
- Los resultados deben complementarse con criterios de negocio y revisión humana.

---

## 16. Conclusiones

El proyecto permitió analizar y modelar el abandono de clientes desde una perspectiva supervisada y no supervisada.

El EDA mostró que las mayores tasas de abandono se asocian con contratos mes a mes, fibra óptica, `Electronic check`, ausencia de soporte técnico, ausencia de seguridad online, menor antigüedad y cargos mensuales más elevados.

Entre los modelos supervisados, la **Regresión Logística** presentó el mejor desempeño, con Accuracy de 80,70%, Recall de 56,68%, F1-Score de 60,92% y ROC-AUC de 84,18%.

A pesar de ser el mejor de los dos modelos comparados, todavía deja sin detectar una proporción relevante de abandonos reales, por lo que se recomienda utilizarlo como apoyo y no como mecanismo automático de decisión.

En aprendizaje no supervisado, K-Means identificó cuatro segmentos. El **Clúster 2**, compuesto por clientes relativamente recientes y con cargos mensuales altos, presentó una tasa de abandono de **48,24%**, convirtiéndose en el segmento de mayor interés para estrategias de retención.

En conjunto, la clasificación permite identificar clientes individuales con señales de riesgo y la segmentación permite detectar grupos con comportamientos diferenciados para orientar acciones más focalizadas.

---

## 17. Matriz de responsabilidades

| Integrante | Responsabilidades principales |
|---|---|
| **Benjamin Luengo** | Comprensión inicial de los datos, revisión de calidad, apoyo en EDA y organización de gráficos e imágenes del proyecto.|
| **Eduardo Rodriguez** | Preparación y transformación de datos, apoyo en modelamiento supervisado y organización de archivos del repositorio. |
| **Diego Iturra** | Evaluación y comparación de modelos, clustering con K-Means, PCA e interpretación de resultados. |
| **Todo el equipo** | Revisión final del notebook, validación del informe, preparación de la presentación y defensa técnica. |

---

## 18. Reproducibilidad

### Librerías principales

```text
pandas
numpy
matplotlib
scikit-learn
```

### Archivos necesarios

```text
EV2_machine_learning.ipynb
Telco_Customer_Churn_Dataset.csv
Informe_Tecnico_EP2_Telco_Churn.md
```

### Reproducibilidad del proyecto

El notebook se encuentra en la carpeta `notebooks/` y el conjunto de datos en `data/`.

La lectura del dataset se realiza mediante:

```python
df = pd.read_csv("../data/Telco_Customer_Churn_Dataset.csv")
```

Esta ruta permite acceder al archivo CSV ubicado en la carpeta `data/` desde el directorio de ejecución `notebooks/`.

Para reproducir el proyecto, es necesario mantener la estructura de carpetas, instalar las librerías utilizadas y ejecutar todas las celdas del notebook en orden, desde un entorno limpio.

---

## 19. Organización del repositorio

El proyecto se organiza de la siguiente manera:

```text
telco-customer-churn-ml/
│
├── README.md
├── .gitignore
│
├── data/
│   └── Telco_Customer_Churn_Dataset.csv
│
├── notebooks/
│   └── EV2_machine_learning.ipynb
│
├── docs/
│   └── Informe_Tecnico_EP2_Telco_Churn.md
│
├── images/
│   └── gráficos utilizados en la presentación
│
└── presentation/
    └── Presentacion_Telco_Churn_EP2.pptx
```

La estructura mantiene separados los datos, el código, la documentación y los materiales de presentación.

El notebook utiliza la ruta relativa `../data/Telco_Customer_Churn_Dataset.csv`, por lo que debe ejecutarse con `notebooks/` como directorio de trabajo para acceder correctamente al conjunto de datos.
