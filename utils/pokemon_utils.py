import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import re

# =============================================================================
# 3.2 ANÁLISE EXPLORATÓRIA (EDA)
# =============================================================================

def plot_missing_data(df):
    """Gera o gráfico de porcentagem de valores nulos."""
    nulos_percentual = (df.isnull().sum() / len(df)) * 100
    # Filtra apenas colunas com dados faltantes e ordena do maior para o menor percentual
    nulos_filtrado = nulos_percentual[nulos_percentual > 0].sort_values(ascending=False)

    if nulos_filtrado.empty:
        print("Nenhum valor nulo encontrado!")
        return

    plt.figure(figsize=(10, 4))
    sns.barplot(x=nulos_filtrado.values, y=nulos_filtrado.index, hue=nulos_filtrado.index, legend=False, palette='Reds_r')
    plt.title('Porcentagem de Dados Faltantes por Coluna', fontsize=14)
    plt.xlabel('Porcentagem de Falhas (%)')
    plt.ylabel('Colunas')

    for index, value in enumerate(nulos_filtrado.values):
        plt.text(value + 0.5, index, f"{value:.1f}%", va='center')
        
    plt.tight_layout()
    plt.show()

def plot_eda_dashboard(df_plot):
    """Gera o dashboard analítico (Boxplot, Scatter, Bar e Heatmap)."""
    sns.set_theme(style="whitegrid")
    
    # Ordem fixa para gerações no gráfico
    ordem_geracoes = ['gen-i', 'gen-ii', 'gen-iii', 'gen-iv', 'gen-v', 'gen-vi', 'gen-vii', 'gen-viii', 'gen-ix']
    
    fig, axes = plt.subplots(2, 2, figsize=(20, 14))

    # 1. BOXPLOT: Distribuição por Geração
    col_gen = 'generation_clean' if 'generation_clean' in df_plot.columns else 'generation'
    if col_gen in df_plot.columns:
        sns.boxplot(x=col_gen, y='base_stat_total', data=df_plot, 
                    order=[g for g in ordem_geracoes if g in df_plot[col_gen].unique()], 
                    ax=axes[0, 0], hue=col_gen, legend=False, palette='coolwarm')
        axes[0, 0].set_title('Distribuição de Poder Total por Geração', fontsize=15)
        axes[0, 0].set_xlabel('Geração')
        axes[0, 0].set_ylabel('Base Stat Total')

    # 2. SCATTER PLOT: Ataque vs Velocidade
    col_leg = 'is_legendary_clean' if 'is_legendary_clean' in df_plot.columns else 'is_legendary'
    if col_leg in df_plot.columns:
        sns.scatterplot(x='attack', y='speed', hue=col_leg, 
                        alpha=0.75, data=df_plot, ax=axes[0, 1], 
                        palette={True: 'red', False: 'dimgray'})
        axes[0, 1].set_title('Estratégia de Batalha: Ataque vs Velocidade', fontsize=15)
        axes[0, 1].legend(title='É Lendário?', loc='upper left')

    # 3. BAR PLOT: Tipagens predominantes
    col_type = 'type_1_clean' if 'type_1_clean' in df_plot.columns else 'type_1'
    if col_type in df_plot.columns:
        type_counts = df_plot[col_type].value_counts()
        cores_dit = {tipo: ('#1f77b4' if i < 3 else 'lightgray') for i, tipo in enumerate(type_counts.index)}
        sns.barplot(x=type_counts.values, y=type_counts.index, ax=axes[1, 0], 
                    hue=type_counts.index, legend=False, palette=cores_dit)
        axes[1, 0].set_title('Quais são as tipagens predominantes? (Top 3 em destaque)', fontsize=15)
        axes[1, 0].set_xlabel('Quantidade de Pokémon')
        axes[1, 0].set_ylabel('Tipo Primário')

    # 4. HEATMAP: Correlação de Atributos
    stats_cols = ['hp', 'attack', 'defense', 'sp_attack', 'sp_defense', 'speed']
    present_stats = [c for c in stats_cols if c in df_plot.columns]
    if present_stats:
        corr_matrix = df_plot[present_stats].corr()
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", ax=axes[1, 1], vmin=-1, vmax=1)
        axes[1, 1].set_title('Mapa de Correlação de Atributos', fontsize=15)

    plt.tight_layout()
    plt.show()

# =============================================================================
# 3.3 LIMPEZA DOS DADOS (Higienização Específica)
# =============================================================================

def clean_numeric_stats(df):
    """3.3.1 - Higieniza colunas numéricas (trata ',' e extrai dígitos)."""
    df_res = df.copy()
    colunas = ['hp', 'attack', 'defense', 'sp_attack', 'sp_defense', 'speed', 'base_stat_total', 
               'height_m', 'weight_kg', 'base_experience', 'capture_rate', 'base_happiness']
    for col in colunas:
        if col in df_res.columns:
            # Remove vírgulas, extrai apenas a parte numérica via regex (ignorando texto) e converte para numérico
            df_res[col] = pd.to_numeric(
                df_res[col].astype(str).str.replace(',', '.').str.extract(r'(\d+\.?\d*)')[0], 
                errors='coerce'
            )
    return df_res

def clean_boolean_columns(df):
    """3.3.2 - Padroniza colunas booleanas para True/False real."""
    df_res = df.copy()
    for col in ['is_legendary', 'is_mythical', 'is_baby']:
        if col in df_res.columns:
            df_res[col] = df_res[col].astype(str).str.upper().str.contains('TRUE')
    return df_res

def clean_simple_text_columns(df):
    """3.3.3 - Limpa caracteres especiais de nomes, tipos e categorias."""
    df_res = df.copy()
    colunas = ['name', 'type_1', 'type_2', 'color', 'shape', 'habitat', 'growth_rate']
    for col in colunas:
        if col in df_res.columns:
            limpo = df_res[col].astype(str).str.replace(r'[^a-zA-Z0-9\-\s]', '', regex=True)
            limpo = limpo.str.strip().str.strip('-').str.capitalize()
            df_res[col] = limpo.replace({'Nan': np.nan, '': np.nan})
    return df_res

def clean_pipe_separated_columns(df):
    """3.3.4 - Limpa e normaliza campos separados por '|'."""
    df_res = df.copy()
    for col in ['abilities', 'hidden_ability', 'egg_groups']:
        if col in df_res.columns:
            # Remove caracteres inválidos, substitui múltiplos hífens/pipes e limpa as bordas
            limpo = (df_res[col].astype(str)
                     .str.replace(r'[^a-zA-Z0-9\-\|]', '', regex=True).str.lower()
                     .str.replace(r'-+', '-', regex=True)
                     .str.replace(r'-\|', '|', regex=True).str.replace(r'\|-', '|', regex=True)
                     .str.strip('-')
                     .replace('nan', np.nan))
            df_res[col] = limpo
    return df_res

def clean_generation_column(df):
    """3.3.5 - Extrai a string de geração (ex: gen-i)."""
    df_res = df.copy()
    if 'generation' in df_res.columns:
        df_res['generation'] = df_res['generation'].astype(str).str.extract(r'(gen-[ixvIXV]+)')[0].str.lower()
    return df_res

def impute_basic_nulls(df):
    """3.3.6 - Preenche nulos críticos com valores padrão."""
    df_res = df.copy()
    if 'type_2' in df_res.columns: df_res['type_2'] = df_res['type_2'].fillna('None')
    if 'name' in df_res.columns: df_res['name'] = df_res['name'].fillna('Unknown')
    return df_res

# =============================================================================
# 3.4 FEATURE ENGINEERING
# =============================================================================

def expand_attributes(df):
    """3.4.1 - Expande strings aglutinadas em colunas separadas."""
    df_fe = df.copy()
    if 'abilities' in df_fe.columns:
        df_hab = df_fe['abilities'].str.split('|', expand=True)
        df_fe['ability_1'] = df_hab[0]
        df_fe['ability_2'] = df_hab[1] if 1 in df_hab.columns else 'None'
        df_fe['ability_2'] = df_fe['ability_2'].fillna('None')
        if 2 in df_hab.columns: df_fe['ability_3'] = df_hab[2].fillna('None')
    
    if 'egg_groups' in df_fe.columns:
        df_egg = df_fe['egg_groups'].str.split('|', expand=True)
        df_fe['egg_group_1'] = df_egg[0]
        df_fe['egg_group_2'] = df_egg[1] if 1 in df_egg.columns else 'None'
        df_fe['egg_group_2'] = df_fe['egg_group_2'].fillna('None')
    return df_fe

def calculate_complex_metrics(df):
    """3.4.1.2 - Calcula IMC, Viés Ofensivo, Potencial Sweeper e Bulk."""
    df_fe = df.copy()
    if 'height_m' in df_fe.columns and 'weight_kg' in df_fe.columns:
        df_fe['bmi'] = np.where(df_fe['height_m'] > 0, df_fe['weight_kg'] / (df_fe['height_m'] ** 2), np.nan)
        df_fe['bmi'] = df_fe['bmi'].fillna(df_fe['bmi'].median())
    
    if all(c in df_fe.columns for c in ['attack', 'sp_attack', 'defense', 'sp_defense']):
        soma_ataque = df_fe['attack'] + df_fe['sp_attack']
        soma_defesa = df_fe['defense'] + df_fe['sp_defense']
        # Viés ofensivo > 1 indica que o Pokémon é mais focado em ataque do que em defesa
        df_fe['offensive_bias'] = np.where(soma_defesa > 0, soma_ataque / soma_defesa, 1)
        if 'speed' in df_fe.columns:
            # Sweeper Potential: métrica de poder de varredura (poder ofensivo x velocidade)
            df_fe['sweeper_potential'] = soma_ataque * df_fe['speed']
            # Physical Bulk: proxy para a resistência física geral do Pokémon
            df_fe['physical_bulk'] = df_fe['hp'] * df_fe['defense']
    return df_fe

def drop_irrelevant_columns(df):
    """3.4.1.3 - Remove colunas ruidosas para modelagem."""
    drop_cols = ['abilities', 'egg_groups', 'sprite_url', 'flavor_text', 'genus', 'evolution_chain_id']
    return df.drop(columns=drop_cols, errors='ignore')

def impute_generation_by_id(df):
    """3.4.2.1 - Imputação de geração via Domain Knowledge (IDs da Pokedex)."""
    df_res = df.copy()
    def _get_gen(id_poke):
        if id_poke <= 151: return 1
        elif id_poke <= 251: return 2
        elif id_poke <= 386: return 3
        elif id_poke <= 493: return 4
        elif id_poke <= 649: return 5
        elif id_poke <= 721: return 6
        elif id_poke <= 809: return 7
        elif id_poke <= 905: return 8
        elif id_poke <= 1025: return 9
        return np.nan

    if 'pokedex_number' in df_res.columns:
        mascara_normais = df_res['pokedex_number'] <= 1025
        df_res.loc[mascara_normais, 'generation'] = df_res.loc[mascara_normais, 'pokedex_number'].apply(_get_gen)
        # Mapeamento do nome base para a geração correspondente (usado para as variações de Pokémon)
        dict_gens = df_res.loc[mascara_normais].set_index('name')['generation'].to_dict()
        mascara_especiais = df_res['pokedex_number'] > 10000
        # Extrai o nome base do Pokémon especial (antes do primeiro hífen)
        nomes_base = df_res.loc[mascara_especiais, 'name'].astype(str).str.split('-').str[0]
        df_res.loc[mascara_especiais, 'generation'] = nomes_base.map(dict_gens)
        df_res['generation'] = df_res['generation'].fillna(0).astype(int)
    return df_res

def encode_features_for_ml(df):
    """3.4.2.2 - Codificação final (Tipos, Labels e One-Hot)."""
    df_ml = df.copy()
    if 'type_1' in df_ml.columns and 'type_2' in df_ml.columns:
        tipos = set(df_ml['type_1'].dropna().unique()).union(set(df_ml['type_2'].dropna().unique()))
        tipos.discard('None') 
        for t in tipos:
            # One-Hot Encoding manual: marca 1 se o Pokémon possui a tipagem (primária ou secundária)
            df_ml[f'type_{t}'] = ((df_ml['type_1'] == t) | (df_ml['type_2'] == t)).astype(int)
        # Remove as colunas originais de tipo após extração das dummies
        df_ml = df_ml.drop(columns=['type_1', 'type_2'], errors='ignore')
    
    for c in ['ability_1', 'ability_2', 'ability_3']:
        if c in df_ml.columns: df_ml[c] = df_ml[c].astype('category').cat.codes
        
    cols_ohe = ['color', 'shape', 'habitat', 'growth_rate']
    return pd.get_dummies(df_ml, columns=[c for c in cols_ohe if c in df_ml.columns], drop_first=True, dtype=int)
