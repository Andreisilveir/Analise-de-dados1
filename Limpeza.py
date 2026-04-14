import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

leitura = pd.read_csv("Analise de dados1/ecommerce_estatistica.csv")

#Analisar se tem algum valor nulo
print(leitura.isnull().sum())

#Analisar se tem alguma incoerencia, como: Desconto negativo, preço negativo, preço MinMax acima de 1, e etc...
print(leitura.describe())

#Analisar se tem algum formato incoerente
print(leitura.info())

# Configuração global de estilo
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

# --- GRÁFICO DE BARRA (Top 10 Marcas por Preço Médio) ---
plt.figure()
top_10_preco = leitura.groupby('Marca')['Preço'].mean().nlargest(10)
sns.barplot(x=top_10_preco.values, y=top_10_preco.index, palette='Blues_d')
plt.title('Top 10 Marcas com Maior Preço Médio')
plt.xlabel('Preço Médio (R$)')
plt.ylabel('Marca')
plt.show()

# --- GRÁFICO DE DISPERSÃO (Preço vs N_Avaliações) ---
plt.figure()
sns.scatterplot(data=leitura, x='Preço', y='N_Avaliações', alpha=0.6, color='coral')
plt.title('Relação: Preço vs Número de Avaliações')
plt.xlabel('Preço (R$)')
plt.ylabel('Total de Avaliações')
plt.show()

# --- HISTOGRAMA (Distribuição de Notas) ---
plt.figure()
sns.histplot(leitura['Nota'], bins=10, kde=False, color='skyblue')
plt.title('Distribuição das Notas dos Produtos')
plt.xlabel('Nota (0 a 5)')
plt.ylabel('Frequência (Quantidade)')
plt.show()

# --- MAPA DE CALOR (Correlação entre Variáveis Numéricas) ---
plt.figure()
# Selecionamos apenas colunas numéricas para a correlação
colunas_num = leitura[['Nota', 'N_Avaliações', 'Desconto', 'Preço', 'Qtd_Vendidos_Cod']]
sns.heatmap(colunas_num.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Mapa de Calor: Correlação entre Variáveis')
plt.show()

# --- GRÁFICO DE PIZZA (Distribuição por Gênero) ---
plt.figure()
genero_counts = leitura['Gênero'].value_counts()
plt.pie(genero_counts, labels=genero_counts.index, autopct='%1.1f%%', startangle=140, colors=sns.color_palette('pastel'))
plt.title('Proporção de Produtos por Gênero')
plt.show()

# --- GRÁFICO DE DENSIDADE (KDE - Distribuição de Descontos) ---
plt.figure()
sns.kdeplot(leitura['Desconto'], fill=True, color="purple")
plt.title('Densidade de Descontos Aplicados')
plt.xlabel('Porcentagem de Desconto')
plt.ylabel('Densidade')
plt.show()

# --- GRÁFICO DE REGRESSÃO (Preço vs Desconto) ---
plt.figure()
sns.regplot(data=leitura, x='Preço', y='Desconto', scatter_kws={'alpha':0.3}, line_kws={'color':'red'})
plt.title('Tendência: Preço vs Desconto')
plt.xlabel('Preço (R$)')
plt.ylabel('Desconto (%)')
plt.show()