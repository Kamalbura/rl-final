#!/usr/bin/env python3
"""
Quick analysis of validation results to understand model performance patterns.
"""

import pandas as pd
import json
import numpy as np

def analyze_validation_results():
    """Analyze the validation results and provide insights."""
    
    # Load validation data
    df = pd.read_csv('validation_results/validation/validation_steps_20250902_050126.csv')
    
    print("="*60)
    print("VALIDATION DATA ANALYSIS")
    print("="*60)
    
    print(f"Total validation steps: {len(df)}")
    print(f"Unique scenarios: {len(df['scenario'].unique())}")
    print(f"Expert agreement rate: {df['agreement'].mean():.3f}")
    print(f"Safety violation rate: {(~df['is_safe']).mean():.3f}")
    
    print("\n" + "="*40)
    print("ACTION DISTRIBUTION")
    print("="*40)
    action_dist = df['agent_action'].value_counts().sort_index()
    action_names = ['No_DDoS', 'XGBoost', 'TST']
    for action, count in action_dist.items():
        print(f"{action_names[action]}: {count} ({count/len(df)*100:.1f}%)")
    
    print("\n" + "="*40)
    print("SCENARIO-WISE PERFORMANCE")
    print("="*40)
    for scenario in df['scenario'].unique():
        scenario_data = df[df['scenario'] == scenario]
        agreement_rate = scenario_data['agreement'].mean()
        safety_rate = scenario_data['is_safe'].mean()
        print(f"{scenario}: {len(scenario_data)} steps, agreement: {agreement_rate:.3f}, safety: {safety_rate:.3f}")
    
    # Parse state information
    print("\n" + "="*40)
    print("THREAT STATE ANALYSIS")
    print("="*40)
    
    # Extract threat information from state strings
    threat_analysis = []
    for idx, row in df.iterrows():
        try:
            state_str = row['state'].replace("'", '"').replace('np.float64(', '').replace(')', '')
            state_dict = json.loads(state_str)
            threat_analysis.append({
                'threat': state_dict['threat'],
                'battery': state_dict['battery'],
                'temperature': state_dict['temperature'],
                'agent_action': row['agent_action'],
                'expert_action': row['expert_action'],
                'agreement': row['agreement'],
                'scenario': row['scenario']
            })
        except:
            continue
    
    threat_df = pd.DataFrame(threat_analysis)
    
    print("THREAT LEVEL DISTRIBUTION:")
    threat_dist = threat_df['threat'].value_counts().sort_index()
    threat_names = ['Normal', 'Confirming', 'Confirmed']
    for threat, count in threat_dist.items():
        print(f"  {threat_names[int(threat)]}: {count}")
    
    print("\nACTION CHOICE BY THREAT LEVEL:")
    for threat in sorted(threat_df['threat'].unique()):
        threat_data = threat_df[threat_df['threat'] == threat]
        action_dist = threat_data['agent_action'].value_counts().sort_index()
        total = len(threat_data)
        print(f"  {threat_names[int(threat)]} threat:")
        for action in [0, 1, 2]:
            count = action_dist.get(action, 0)
            print(f"    {action_names[action]}: {count} ({count/total*100:.1f}%)")
    
    print("\nBATTERY vs THREAT ANALYSIS:")
    for threat in sorted(threat_df['threat'].unique()):
        threat_data = threat_df[threat_df['threat'] == threat]
        avg_battery = threat_data['battery'].mean()
        print(f"  {threat_names[int(threat)]}: avg battery {avg_battery:.1f}%")
        
        # Battery ranges
        low_battery = threat_data[threat_data['battery'] < 30]
        high_battery = threat_data[threat_data['battery'] > 60]
        print(f"    Low battery (<30%): {len(low_battery)} cases")
        print(f"    High battery (>60%): {len(high_battery)} cases")
    
    print("\nTEMPERATURE vs THREAT ANALYSIS:")
    for threat in sorted(threat_df['threat'].unique()):
        threat_data = threat_df[threat_df['threat'] == threat]
        avg_temp = threat_data['temperature'].mean()
        print(f"  {threat_names[int(threat)]}: avg temperature {avg_temp:.1f}°C")
        
        # Temperature ranges
        hot_temp = threat_data[threat_data['temperature'] > 60]
        critical_temp = threat_data[threat_data['temperature'] > 75]
        print(f"    Hot (>60°C): {len(hot_temp)} cases")
        print(f"    Critical (>75°C): {len(critical_temp)} cases")

if __name__ == "__main__":
    analyze_validation_results()