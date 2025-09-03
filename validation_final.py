#!/usr/bin/env python3
"""
UAV Cybersecurity RL System - FINAL Validation & Threat-State Analysis Suite

Enhancements over validate_and_plot.py:
- Integrates threat-state conditioned battery & temperature analysis directly during validation
- Saves unified metrics JSON including per-threat distributions & cross-tabs
- Adds dedicated threat-state visualization dashboard

Author: UAV RL Team
Date: September 2, 2025
Version: 2.0 (final validation)
"""
import os, sys, json, argparse, logging, signal, traceback
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

try:
    from agents.q_learning_agent import WarmStartQLearningAgent
    from agents.expert_policy import ExpertPolicy
    from environment.thermal_simulator import TunableUAVSimulator
    from utils.state_discretizer import StateDiscretizer
    from validation.safety_validator import SafetyValidator
except ImportError as e:
    print(f"Import error: {e}. Run from project root.")
    sys.exit(1)

class FinalValidationSuite:
    def __init__(self, model_path: str, output_dir: str):
        self.model_path = model_path
        self.output_dir = Path(output_dir)
        self.validation_dir = self.output_dir / 'validation'
        self.plots_dir = self.output_dir / 'plots'
        self.validation_dir.mkdir(parents=True, exist_ok=True)
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        self._setup_logging()
        self.colors = {
            'primary':'#2E5266','secondary':'#4A90A4','accent':'#85C1E9',
            'success':'#27AE60','warning':'#F39C12','danger':'#E74C3C',
            'gray_light':'#BDC3C7','gray_dark':'#34495E','background':'#F8F9FA'
        }
        self.action_labels = ['No_DDoS','XGBoost','TST']
        self.battery_levels = ['0-20%','21-40%','41-60%','61-80%','81-100%']
        self.temp_levels = ['Safe','Warning','Critical']
        self.threat_levels = ['Normal','Confirming','Confirmed']
        self.validation_data = []
        self.state_combination_results = {}
        self.scenario_results = {}
        self._setup_plot_style()
        self._init_components()
        signal.signal(signal.SIGINT, self._signal_handler)

    def _setup_logging(self):
        log_file = self.validation_dir / 'validation_final.log'
        logging.basicConfig(level=logging.INFO,
                            format='%(asctime)s - %(levelname)s - %(message)s',
                            handlers=[logging.FileHandler(log_file), logging.StreamHandler()])
        self.logger = logging.getLogger('validation_final')

    def _setup_plot_style(self):
        plt.style.use('default')
        sns.set_palette([self.colors['primary'], self.colors['secondary'], self.colors['accent']])
        plt.rcParams.update({'figure.dpi':300,'savefig.dpi':300,'axes.grid':True,'grid.alpha':0.3,'font.size':11})

    def _init_components(self):
        try:
            with open('config/simulator_params.json','r') as f:
                sim_cfg = json.load(f)
            self.simulator = TunableUAVSimulator(sim_cfg)
            self.expert_policy = ExpertPolicy('config/expert_policy.json')
            self.state_discretizer = StateDiscretizer()
            self.safety_validator = SafetyValidator(self.simulator)
            self.agent = WarmStartQLearningAgent(self.expert_policy, self.state_discretizer)
            self.agent.load(self.model_path)
            self.logger.info('Components initialized')
        except Exception as e:
            self.logger.error(f'Initialization failed: {e}')
            raise

    def _signal_handler(self, *_):
        self.logger.info('Interrupt received, saving partial results...')
        if self.validation_data:
            metrics = self._compute_metrics()  # ensure metrics for partial
            self._save_all(metrics)
        sys.exit(0)

    # --- Core Validation ---
    def run(self):
        self.logger.info('Starting FINAL validation suite...')
        self._validate_state_space()
        self._run_scenarios()
        metrics = self._compute_metrics()
        self._save_all(metrics)
        self._plot_all(metrics)
        self.logger.info('FINAL validation complete.')
        return metrics

    def _validate_state_space(self):
        total = len(self.battery_levels)*len(self.temp_levels)*len(self.threat_levels)
        for i, b in enumerate(self.battery_levels):
            for j, t in enumerate(self.temp_levels):
                for k, th in enumerate(self.threat_levels):
                    combo_id = f'{b}|{t}|{th}'
                    state = self._representative_state(b,t,th)
                    agent_action = self.agent.get_action(state, training=False)
                    expert_action = self.expert_policy.get_action(state)
                    safe = self._is_safe(state, agent_action)
                    res = {'battery_level':b,'temp_level':t,'threat_level':th,'state':state,
                           'agent_action':agent_action,'expert_action':expert_action,
                           'agreement':agent_action==expert_action,'is_safe':safe}
                    self.state_combination_results[combo_id]=res
            progress = ( (i+1)/len(self.battery_levels) )*100
            self.logger.info(f'State combinations progress: {progress:.1f}%')

    def _representative_state(self,batt,temp,threat):
        battery_map={'0-20%':15.0,'21-40%':30.0,'41-60%':50.0,'61-80%':70.0,'81-100%':90.0}
        temp_map={'Safe':45.0,'Warning':65.0,'Critical':82.0}
        threat_map={'Normal':0,'Confirming':1,'Confirmed':2}
        return {'battery':battery_map[batt],'temperature':temp_map[temp],'threat':threat_map[threat],
                'cpu_usage':50.0,'time_since_tst':300.0,'power':4.0}

    def _run_scenarios(self):
        scenarios=['normal_operation','hot_conditions','low_battery','critical_battery',
                   'continuous_threat','temperature_stress','mixed_conditions','tst_recovery']
        for s in scenarios:
            self.logger.info(f'Running scenario: {s}')
            self.scenario_results[s]=self._run_scenario(s)

    def _run_scenario(self,name,episodes: int=5):
        data={'name':name,'episodes':[],'summary':{}}
        for ep in range(episodes):
            data['episodes'].append(self._run_episode(name, ep))
        data['summary']=self._scenario_summary(data['episodes'])
        return data

    def _initial_state(self,scenario):
        base={'battery':80.0,'temperature':45.0,'threat':0,'cpu_usage':30.0,'time_since_tst':300.0,'power':3.0}
        mods={
            'normal_operation':{},'hot_conditions':{'temperature':70.0},'low_battery':{'battery':25.0},
            'critical_battery':{'battery':15.0},'continuous_threat':{'threat':2},
            'temperature_stress':{'temperature':80.0},'mixed_conditions':{'battery':30.0,'temperature':75.0,'threat':1},
            'tst_recovery':{'time_since_tst':50.0,'temperature':55.0}
        }
        base.update(mods.get(scenario,{}))
        return base

    def _run_episode(self,scenario,ep_num):
        self.simulator.reset()
        init_state=self._initial_state(scenario)
        self.simulator.current_temp=init_state['temperature']
        self.simulator.current_battery=init_state['battery']
        self.simulator.current_cpu_usage=init_state['cpu_usage']
        self.simulator.current_power=init_state['power']
        self.simulator.time_since_tst=init_state['time_since_tst']
        state=init_state.copy()
        steps=[]; agreements=0; violations=0; total_reward=0
        max_steps=100
        for step in range(max_steps):
            # Threat dynamics
            if scenario=='continuous_threat':
                state['threat']=2
            elif scenario=='mixed_conditions':
                state['threat']=(step//20)%3
            elif 30<step<60:
                state['threat']=min(state['threat']+1,2)
            agent_action=self.agent.get_action(state,training=False)
            expert_action=self.expert_policy.get_action(state)
            safe=self._is_safe(state,agent_action)
            agree=agent_action==expert_action
            if agree: agreements+=1
            if not safe: violations+=1
            sim=self.simulator.simulate_step(agent_action)
            next_state={'temperature':sim['temperature'],'battery':sim['battery'],'threat':state['threat'],
                        'cpu_usage':sim['cpu_usage'],'time_since_tst':sim.get('time_since_tst',state['time_since_tst']),
                        'power':sim['power_consumption']}
            reward=1.0 + (0.5 if agree else 0) - (10.0 if not safe else 0)
            total_reward+=reward
            done = sim['temperature']>85.0 or sim['battery']<5.0 or step>=max_steps-1
            step_record={'step':step,'state':state.copy(),'agent_action':agent_action,'expert_action':expert_action,
                         'reward':reward,'is_safe':safe,'agreement':agree,'next_state':next_state.copy(),'done':done}
            steps.append(step_record)
            self.validation_data.append({**step_record,'scenario':scenario,'episode':ep_num,'timestamp':datetime.now().isoformat()})
            state=next_state
            if done: break
        summary={'total_steps':len(steps), 'expert_agreement_rate':agreements/max(1,len(steps)),
                 'safety_violation_rate':violations/max(1,len(steps)), 'total_reward':total_reward,
                 'max_temperature':max(s['state']['temperature'] for s in steps),
                 'min_battery':min(s['state']['battery'] for s in steps), 'final_state':state}
        return {'episode':ep_num,'scenario':scenario,'steps':steps,'summary':summary}

    def _is_safe(self,state,action):
        if state['temperature']>85.0: return False
        if state['battery']<10.0: return False
        if action==2: # TST
            if state['temperature']>80.0 or state['battery']<30.0: return False
            if state['time_since_tst']<120.0: return False
        return True

    def _scenario_summary(self,episodes):
        if not episodes: return {}
        s=[e['summary'] for e in episodes]
        return {
            'episodes_count':len(episodes),
            'avg_expert_agreement':float(np.mean([x['expert_agreement_rate'] for x in s])),
            'avg_safety_violation_rate':float(np.mean([x['safety_violation_rate'] for x in s])),
            'avg_total_reward':float(np.mean([x['total_reward'] for x in s])),
            'max_temperature_observed':float(max(x['max_temperature'] for x in s)),
            'min_battery_observed':float(min(x['min_battery'] for x in s)),
            'avg_steps':float(np.mean([x['total_steps'] for x in s])),
            'success_rate':float(sum(1 for x in s if x['safety_violation_rate']==0)/len(s))
        }

    # --- Metrics & Analysis ---
    def _compute_metrics(self):
        metrics={'timestamp':datetime.now().isoformat(),'model_path':self.model_path,
                 'validation_summary':{},'state_combination_analysis':{},'scenario_analysis':{},
                 'safety_metrics':{},'performance_metrics':{},'threat_state_analysis':{}}
        total_steps=len(self.validation_data)
        agreements=sum(1 for d in self.validation_data if d['agreement'])
        violations=sum(1 for d in self.validation_data if not d['is_safe'])
        metrics['validation_summary']={
            'total_validation_steps':total_steps,
            'total_scenarios_tested':len(self.scenario_results),
            'total_state_combinations_tested':len(self.state_combination_results),
            'overall_expert_agreement':agreements/max(1,total_steps),
            'overall_safety_violation_rate':violations/max(1,total_steps)
        }
        combo_agree=sum(1 for r in self.state_combination_results.values() if r['agreement'])
        combo_unsafe=sum(1 for r in self.state_combination_results.values() if not r['is_safe'])
        total_combo=len(self.state_combination_results)
        # Capture detailed disagreements for traceability
        disagreement_details=[]
        for combo_id, r in self.state_combination_results.items():
            if not r['agreement']:
                disagreement_details.append({
                    'combination_id': combo_id,
                    'battery_level': r['battery_level'],
                    'temp_level': r['temp_level'],
                    'threat_level': r['threat_level'],
                    'agent_action': int(r['agent_action']),
                    'expert_action': int(r['expert_action'])
                })
        metrics['state_combination_analysis']={
            'total_combinations':total_combo,
            'expert_agreement_rate':combo_agree/max(1,total_combo),
            'safety_violation_rate':combo_unsafe/max(1,total_combo),
            'combinations_with_disagreement':total_combo - combo_agree,
            'combinations_with_violations':combo_unsafe,
            'disagreement_details':disagreement_details
        }
        metrics['scenario_analysis']={k:v['summary'] for k,v in self.scenario_results.items()}
        if self.validation_data:
            temps=[d['state']['temperature'] for d in self.validation_data]
            batts=[d['state']['battery'] for d in self.validation_data]
            metrics['safety_metrics']={
                'max_temperature_observed':max(temps),'min_battery_observed':min(batts),
                'temperature_violations':sum(1 for t in temps if t>85.0),
                'battery_violations':sum(1 for b in batts if b<10.0),
                'critical_temperature_approaches':sum(1 for t in temps if t>80.0),
                'critical_battery_approaches':sum(1 for b in batts if b<20.0)
            }
            rewards=[d['reward'] for d in self.validation_data]
            action_dist={f'action_{a}':sum(1 for d in self.validation_data if d['agent_action']==a) for a in [0,1,2]}
            metrics['performance_metrics']={
                'avg_reward':float(np.mean(rewards)),'total_reward':float(np.sum(rewards)),
                'reward_std':float(np.std(rewards)),'action_distribution':action_dist
            }
        # Threat-state extended analysis
        metrics['threat_state_analysis']=self._compute_threat_state_analysis()
        return metrics

    def _compute_threat_state_analysis(self):
        if not self.validation_data:
            return {}
        threat_name={0:'Normal',1:'Confirming',2:'Confirmed'}
        # Helper bucket functions
        def battery_bucket(x):
            if x<20: return '0-20%'
            if x<40: return '21-40%'
            if x<60: return '41-60%'
            if x<80: return '61-80%'
            return '81-100%'
        def temp_bucket(x):
            if x<60: return 'Safe'
            if x<80: return 'Warning'
            return 'Critical'
        rows=[]
        for d in self.validation_data:
            s=d['state']
            rows.append({
                'threat_idx':s['threat'],'threat':threat_name.get(s['threat'],str(s['threat'])),
                'battery':s['battery'],'temperature':s['temperature'],
                'battery_bucket':battery_bucket(s['battery']),
                'temperature_bucket':temp_bucket(s['temperature']),
                'agent_action':d['agent_action'],'agreement':d['agreement'],'is_safe':d['is_safe']
            })
        df=pd.DataFrame(rows)
        per_threat={}
        for th, g in df.groupby('threat'):
            total=len(g)
            act_counts=g['agent_action'].value_counts().to_dict()
            act_pct={self.action_labels[k]:act_counts.get(k,0)/total for k in [0,1,2]}
            batt_dist=g['battery_bucket'].value_counts(normalize=True).to_dict()
            temp_dist=g['temperature_bucket'].value_counts(normalize=True).to_dict()
            per_threat[th]={
                'count':int(total),
                'avg_battery':float(g['battery'].mean()),'battery_std':float(g['battery'].std()),
                'avg_temperature':float(g['temperature'].mean()),'temperature_std':float(g['temperature'].std()),
                'action_distribution_pct':act_pct,
                'battery_bucket_distribution_pct':batt_dist,
                'temperature_bucket_distribution_pct':temp_dist,
                'agreement_rate':float(g['agreement'].mean()),
                'safety_rate':float(g['is_safe'].mean()),
                'min_battery':float(g['battery'].min()),'max_battery':float(g['battery'].max()),
                'min_temperature':float(g['temperature'].min()),'max_temperature':float(g['temperature'].max())
            }
        # Cross-tabs (normalized within threat)
        pivot_batt=df.pivot_table(index='threat',columns='battery_bucket',values='agent_action',aggfunc='count',fill_value=0)
        pivot_temp=df.pivot_table(index='threat',columns='temperature_bucket',values='agent_action',aggfunc='count',fill_value=0)
        pivot_batt_norm=pivot_batt.div(pivot_batt.sum(axis=1),axis=0)
        pivot_temp_norm=pivot_temp.div(pivot_temp.sum(axis=1),axis=0)
        return {
            'per_threat':per_threat,
            'battery_bucket_distribution_matrix':pivot_batt_norm.round(4).to_dict(),
            'temperature_bucket_distribution_matrix':pivot_temp_norm.round(4).to_dict()
        }

    # --- Saving & Plotting ---
    def _save_all(self, metrics: Dict[str,Any]):
        ts=datetime.now().strftime('%Y%m%d_%H%M%S')
        # CSV
        if self.validation_data:
            df=pd.DataFrame(self.validation_data)
            df.to_csv(self.validation_dir / f'validation_steps_final_{ts}.csv', index=False)
        # State combos
        with open(self.validation_dir / f'state_combinations_final_{ts}.json','w') as f:
            json.dump(self.state_combination_results,f,indent=2,default=str)
        # Scenarios
        with open(self.validation_dir / f'scenario_results_final_{ts}.json','w') as f:
            json.dump(self.scenario_results,f,indent=2,default=str)
        # Metrics (unified)
        with open(self.validation_dir / f'metrics_final_{ts}.json','w') as f:
            json.dump(metrics,f,indent=2)
        # Per-threat CSV summary for quick external consumption
        tsa=metrics.get('threat_state_analysis',{}).get('per_threat',{})
        if tsa:
            rows=[]
            for th, d in tsa.items():
                row={'threat':th,
                     'count':d['count'],
                     'avg_battery':d['avg_battery'],
                     'battery_std':d['battery_std'],
                     'avg_temperature':d['avg_temperature'],
                     'temperature_std':d['temperature_std'],
                     'agreement_rate':d['agreement_rate'],
                     'safety_rate':d['safety_rate']}
                # Flatten action distribution
                for a, v in d['action_distribution_pct'].items():
                    row[f'action_{a}_pct']=v
                rows.append(row)
            pd.DataFrame(rows).to_csv(self.validation_dir / f'threat_state_summary_{ts}.csv', index=False)
        self.logger.info('All results saved (CSV, state combos, scenarios, metrics).')

    def _plot_all(self, metrics: Dict[str,Any]):
        self._plot_threat_state_dashboard(metrics['threat_state_analysis'])
        # Reuse selective plots from original suite for continuity
        self._plot_agreement_heat(metrics)
        self._plot_action_vs_threat(metrics['threat_state_analysis'])

    def _plot_threat_state_dashboard(self, analysis: Dict[str,Any]):
        if not analysis: return
        per=analysis['per_threat']
        threats=list(per.keys())
        fig, axes=plt.subplots(2,2,figsize=(16,12))
        fig.suptitle('Threat-State Battery & Temperature Performance', fontsize=16, fontweight='bold', color=self.colors['primary'])
        # Avg battery
        avg_batt=[per[t]['avg_battery'] for t in threats]
        std_batt=[per[t]['battery_std'] for t in threats]
        axes[0,0].bar(threats, avg_batt, yerr=std_batt, color=self.colors['secondary'], alpha=0.8, capsize=5)
        axes[0,0].set_ylabel('Battery (%)'); axes[0,0].set_title('Average Battery by Threat')
        axes[0,0].grid(True,alpha=0.3)
        # Avg temperature
        avg_temp=[per[t]['avg_temperature'] for t in threats]
        std_temp=[per[t]['temperature_std'] for t in threats]
        axes[0,1].bar(threats, avg_temp, yerr=std_temp, color=self.colors['accent'], alpha=0.8, capsize=5)
        axes[0,1].set_ylabel('Temperature (°C)'); axes[0,1].set_title('Average Temperature by Threat')
        axes[0,1].grid(True,alpha=0.3)
        # Battery bucket distribution stacked
        batt_buckets=self.battery_levels
        bottom=np.zeros(len(threats))
        for bucket in batt_buckets:
            vals=[per[t]['battery_bucket_distribution_pct'].get(bucket,0) for t in threats]
            axes[1,0].bar(threats, vals, bottom=bottom, label=bucket)
            bottom+=np.array(vals)
        axes[1,0].set_ylim(0,1.05); axes[1,0].set_title('Battery Bucket Distribution by Threat')
        axes[1,0].set_ylabel('Proportion'); axes[1,0].legend(frameon=False, fontsize=8)
        # Temperature bucket distribution stacked
        temp_buckets=self.temp_levels
        bottom=np.zeros(len(threats))
        for bucket in temp_buckets:
            vals=[per[t]['temperature_bucket_distribution_pct'].get(bucket,0) for t in threats]
            axes[1,1].bar(threats, vals, bottom=bottom, label=bucket)
            bottom+=np.array(vals)
        axes[1,1].set_ylim(0,1.05); axes[1,1].set_title('Temperature Bucket Distribution by Threat')
        axes[1,1].set_ylabel('Proportion'); axes[1,1].legend(frameon=False, fontsize=8)
        for ax in axes.flat: ax.set_xlabel('Threat State')
        plt.tight_layout()
        plt.savefig(self.plots_dir / 'threat_state_dashboard.png', dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_agreement_heat(self, metrics: Dict[str,Any]):
        # Agreement vs safety heat similar to original but condensed
        safety = np.zeros((len(self.battery_levels), len(self.temp_levels)))
        disagree = np.zeros((len(self.battery_levels), len(self.temp_levels)))
        for combo_id,res in self.state_combination_results.items():
            b,t,th = combo_id.split('|')
            bi=self.battery_levels.index(b); ti=self.temp_levels.index(t)
            if not res['is_safe']: safety[bi,ti]+=1
            if not res['agreement']: disagree[bi,ti]+=1
        safety/=3; disagree/=3
        fig, axes = plt.subplots(1,2, figsize=(14,6))
        fig.suptitle('State Space Safety & Disagreement Heatmaps', fontsize=14, fontweight='bold')
        sns.heatmap(safety, annot=True, fmt='.2f', cmap='Reds', xticklabels=self.temp_levels, yticklabels=self.battery_levels, ax=axes[0], cbar_kws={'label':'Violation Rate'})
        axes[0].set_title('Safety Violations'); axes[0].set_xlabel('Temperature Level'); axes[0].set_ylabel('Battery Level')
        sns.heatmap(disagree, annot=True, fmt='.2f', cmap='Blues', xticklabels=self.temp_levels, yticklabels=self.battery_levels, ax=axes[1], cbar_kws={'label':'Disagreement Rate'})
        axes[1].set_title('Expert Disagreement'); axes[1].set_xlabel('Temperature Level'); axes[1].set_ylabel('Battery Level')
        plt.tight_layout(); plt.savefig(self.plots_dir / 'state_space_heatmaps_final.png', dpi=300, bbox_inches='tight'); plt.close()

    def _plot_action_vs_threat(self, analysis: Dict[str,Any]):
        if not analysis: return
        per=analysis['per_threat']
        threats=list(per.keys())
        # Action distribution stacked bar
        fig, ax = plt.subplots(figsize=(10,6))
        bottom=np.zeros(len(threats))
        for action_label in self.action_labels:
            vals=[per[t]['action_distribution_pct'][action_label] for t in threats]
            ax.bar(threats, vals, bottom=bottom, label=action_label)
            bottom+=np.array(vals)
        ax.set_ylim(0,1.05); ax.set_ylabel('Proportion'); ax.set_title('Agent Action Distribution per Threat State')
        ax.legend(frameon=False)
        plt.tight_layout(); plt.savefig(self.plots_dir / 'action_distribution_threat_final.png', dpi=300, bbox_inches='tight'); plt.close()


def main():
    parser=argparse.ArgumentParser(description='FINAL Validation & Threat-State Analysis')
    parser.add_argument('--model_path', required=True, help='Path to trained model JSON (Q-learning)')
    parser.add_argument('--outdir', default='validation_results_final', help='Output directory')
    args=parser.parse_args()
    try:
        suite=FinalValidationSuite(args.model_path, args.outdir)
        metrics=suite.run()
        # Console summary focusing on threat-state battery & temperature
        tsa=metrics['threat_state_analysis'].get('per_threat',{})
        print('\n'+'='*90) ; print('THREAT-STATE PERFORMANCE SUMMARY (Battery & Temperature Focus)') ; print('='*90)
        for th,data in tsa.items():
            print(f"{th}: count={data['count']}, avg_battery={data['avg_battery']:.1f}%, avg_temp={data['avg_temperature']:.1f}°C, agreement={data['agreement_rate']:.1%}")
            print(f"  Battery buckets: "+', '.join(f"{k} {v*100:.1f}%" for k,v in sorted(data['battery_bucket_distribution_pct'].items())))
            print(f"  Temp buckets: "+', '.join(f"{k} {v*100:.1f}%" for k,v in sorted(data['temperature_bucket_distribution_pct'].items())))
            ad=data['action_distribution_pct']
            print(f"  Actions: "+', '.join(f"{k} {v*100:.1f}%" for k,v in ad.items()))
        print('='*90)
        print(f"Metrics & plots saved under: {args.outdir}")
        print('Key plots: threat_state_dashboard.png, action_distribution_threat_final.png, state_space_heatmaps_final.png')
    except Exception as e:
        print(f'Validation failed: {e}')
        traceback.print_exc(); sys.exit(1)

if __name__=='__main__':
    main()
