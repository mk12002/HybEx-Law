import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# --- Configuration ---
INPUT_FILE = r'D:\Code_stuff\HybEx-Law\multi_domain_legal\results\ablation_study\final.csv'
OUTPUT_DIR = 'ablation_graphs'
# ---------------------

def load_and_prep_data(filename: str) -> pd.DataFrame:
    """Loads and cleans the CSV data for plotting."""
    try:
        df = pd.read_csv(filename)
    except FileNotFoundError:
        print(f"Error: Input file not found.")
        print(f"Please make sure '{filename}' is in the same directory as this script.")
        return None

    # 1. Clean metric columns (convert "98.48%" to 0.9848)
    for col in ['Accuracy', 'Precision', 'Recall']:
        if df[col].dtype == 'object':
            df[col] = df[col].str.replace('%', '').astype(float) / 100.0
            
    # 2. Rename F1-Score for easier access
    if 'F1-Score' in df.columns:
        df = df.rename(columns={'F1-Score': 'F1'})

    # 3. Engineer 'num_models' column
    df['num_models'] = df['Models Used'].str.count(r'\+') + 1

    # 4. Engineer 'Ensemble Type' column
    def categorize_ensemble(row):
        if row['num_models'] == 1:
            return 'Single Model'
        if 'Prolog' in row['Models Used']:
            return 'Hybrid Ensemble'
        else:
            return 'Neural-Only Ensemble'
    
    df['Ensemble Type'] = df.apply(categorize_ensemble, axis=1)
    
    # 5. Simplify 'Models Used' for single models
    df['Model Name'] = df['Models Used'].str.replace(r' \(only\)', '', regex=True)
    
    print(f"Successfully loaded and processed '{filename}'.")
    return df

def plot_graph_1(df, path):
    """Graph 1: Overall Performance Leaderboard (Grouped Bar Chart)"""
    print("Generating Graph 1: Overall Performance Leaderboard...")
    df_melted = df.melt(id_vars='Combination', 
                        value_vars=['Accuracy', 'Precision', 'Recall', 'F1'], 
                        var_name='Metric', 
                        value_name='Score')

    plt.figure(figsize=(20, 10))
    sns.barplot(data=df_melted, x='Combination', y='Score', hue='Metric')
    plt.title('Overall Performance Leaderboard by Metric', fontsize=20, weight='bold')
    plt.xlabel('Model Combination', fontsize=14)
    plt.ylabel('Score', fontsize=14)
    plt.xticks(rotation=90, ha='center')
    plt.legend(title='Metric', fontsize=12, loc='lower left')
    plt.ylim(0.8, 1.0)  # Zoom in on the top scores
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(path, '1_overall_performance_leaderboard.png'))
    plt.close()

def plot_graph_2(df, path):
    """Graph 2: Precision-Recall Trade-off (Scatter Plot)"""
    print("Generating Graph 2: Precision-Recall Trade-off...")
    plt.figure(figsize=(14, 9))
    sns.scatterplot(
        data=df, 
        x='Recall', 
        y='Precision', 
        size='F1', 
        hue='Combination', 
        sizes=(100, 1000), 
        legend='brief',
        alpha=0.7
    )
    plt.title('Precision-Recall Trade-off', fontsize=20, weight='bold')
    plt.xlabel('Recall', fontsize=14)
    plt.ylabel('Precision', fontsize=14)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(path, '2_precision_recall_tradeoff.png'))
    plt.close()

def plot_graph_3(df, path):
    """Graph 3: Single-Component Ranking (Bar Chart)"""
    print("Generating Graph 3: Single-Component Ranking...")
    df_single = df[df['Ensemble Type'] == 'Single Model'].sort_values('F1', ascending=False)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df_single, x='Model Name', y='F1', palette='viridis', order=df_single['Model Name'])
    plt.title('Single Model Component Performance (Ranked by F1)', fontsize=18, weight='bold')
    plt.xlabel('Model', fontsize=14)
    plt.ylabel('F1-Score', fontsize=14)
    plt.ylim(0.8, 1.0)
    plt.tight_layout()
    plt.savefig(os.path.join(path, '3_single_component_ranking.png'))
    plt.close()

def plot_graph_4(df, path):
    """Graph 4: The "Enhanced BERT" Impact (Grouped Bar Chart)"""
    print("Generating Graph 4: The 'Enhanced BERT' Impact...")
    pairs = {
        'GNN': ('02_gnn', '11_gnn_bert'),
        'Domain': ('03_domain', '07_domain_bert'),
        'Eligibility': ('04_eligibility', '08_eligibility_bert')
    }
    plot_data = []
    
    for base_name, (base_combo_id, plus_bert_combo_id) in pairs.items():
        base_f1 = df[df['Combination'] == base_combo_id]['F1'].values[0]
        bert_f1 = df[df['Combination'] == plus_bert_combo_id]['F1'].values[0]
        
        plot_data.append({'Base Model': base_name, 'Combination Type': 'Base Model Only', 'F1-Score': base_f1})
        plot_data.append({'Base Model': base_name, 'Combination Type': '+ Enhanced BERT', 'F1-Score': bert_f1})

    df_plot = pd.DataFrame(plot_data)

    plt.figure(figsize=(12, 7))
    sns.barplot(data=df_plot, x='Base Model', y='F1-Score', hue='Combination Type', palette='pastel')
    plt.title('Impact of Adding "Enhanced Legal BERT"', fontsize=20, weight='bold')
    plt.xlabel('Base Model Component', fontsize=14)
    plt.ylabel('F1-Score', fontsize=14)
    plt.legend(title='Combination Type', loc='lower right')
    plt.ylim(0.85, 1.0)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(path, '4_enhanced_bert_impact.png'))
    plt.close()

def plot_graph_5(df, path):
    """Graph 5: The Value of Prolog (Bar Chart)"""
    print("Generating Graph 5: The Value of Prolog...")
    plot_data = [
        {'Combination': 'All Neural (Rank 2)', 'F1': df[df['Combination'] == '15_all_neural']['F1'].values[0]},
        {'Combination': 'Full Ensemble (Rank 1)', 'F1': df[df['Combination'] == '17_full_ensemble']['F1'].values[0]}
    ]
    df_plot = pd.DataFrame(plot_data)

    plt.figure(figsize=(8, 6))
    ax = sns.barplot(data=df_plot, x='Combination', y='F1', palette='muted')
    plt.title('The Value of Adding Prolog to the Neural Ensemble', fontsize=16, weight='bold')
    plt.xlabel('Ensemble', fontsize=14)
    plt.ylabel('F1-Score', fontsize=14)
    plt.ylim(0.98, 0.986)  # Zoom in to show the small, important difference
    
    for p in ax.patches:
        ax.annotate(f'{p.get_height():.4f}', (p.get_x() + p.get_width() / 2., p.get_height()), 
                    ha='center', va='center', fontsize=12, color='black', xytext=(0, 5), 
                    textcoords='offset points')
        
    plt.tight_layout()
    plt.savefig(os.path.join(path, '5_value_of_prolog.png'))
    plt.close()

def plot_graph_6(df, path):
    """Graph 6: Performance vs. Complexity (Scatter/Box Plot)"""
    print("Generating Graph 6: Performance vs. Complexity...")
    plt.figure(figsize=(12, 8))
    
    # Draw a boxplot to show the distribution at each complexity level
    sns.boxplot(data=df, x='num_models', y='F1', 
                showfliers=False, boxprops=dict(alpha=.3), color='lightgray')
    # Overlay a stripplot (jittered scatter) to show all individual points
    sns.stripplot(data=df, x='num_models', y='F1', 
                  jitter=0.1, alpha=0.7, size=8, hue='Ensemble Type', legend=True)

    plt.title('Performance vs. Ensemble Complexity', fontsize=20, weight='bold')
    plt.xlabel('Number of Models in Ensemble', fontsize=14)
    plt.ylabel('F1-Score', fontsize=14)
    plt.legend(title='Ensemble Type', loc='lower right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(path, '6_performance_vs_complexity.png'))
    plt.close()

def plot_graph_7(df, path):
    """Graph 7: Model Type Performance (Box Plot)"""
    print("Generating Graph 7: Model Type Performance...")
    order = ['Single Model', 'Neural-Only Ensemble', 'Hybrid Ensemble']
    
    plt.figure(figsize=(10, 7))
    sns.boxplot(data=df, x='Ensemble Type', y='F1', order=order, palette='deep')
    # Overlay the individual points
    sns.stripplot(data=df, x='Ensemble Type', y='F1', 
                  order=order, 
                  color='black', alpha=0.5, jitter=0.1)
                  
    plt.title('Performance by Ensemble Type', fontsize=20, weight='bold')
    plt.xlabel('Ensemble Type', fontsize=14)
    plt.ylabel('F1-Score', fontsize=14)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(path, '7_model_type_performance.png'))
    plt.close()

def main():
    """Main function to run the script."""
    
    # Set a consistent, professional style for all plots
    sns.set_theme(style="whitegrid", context="talk")
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Load and prepare data
    data = load_and_prep_data(INPUT_FILE)
    
    if data is not None:
        # Generate all graphs
        plot_graph_1(data, OUTPUT_DIR)
        plot_graph_2(data, OUTPUT_DIR)
        plot_graph_3(data, OUTPUT_DIR)
        plot_graph_4(data, OUTPUT_DIR)
        plot_graph_5(data, OUTPUT_DIR)
        plot_graph_6(data, OUTPUT_DIR)
        plot_graph_7(data, OUTPUT_DIR)
        
        print(f"\nAll 7 graphs have been generated and saved to the '{OUTPUT_DIR}' folder.")

if __name__ == "__main__":
    main()