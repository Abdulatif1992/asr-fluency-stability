#!/usr/bin/env python3
"""STEP 12.4: exact speaker-level paired sign-flip tests + Holm family-wise adjustment.

Reads archived, unmodified out-of-fold predictions. No ASR, feature fitting,
model fitting, or access to labels outside the original 400-audio pilot.
The four contrasts and multiple-test family were chosen after viewing prior results;
all inference remains exploratory, NOT prospective confirmation.
"""
from __future__ import annotations
import hashlib, json, zipfile
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import trim_mean

ROOT = Path(__file__).resolve().parent.parent / 'reproducibility' / 'step12_4'
SEED = 20261102
NBOOT = 50000
CONTRASTS = (
    ('C1_minus_A','C1','A','MAE'),
    ('C2_minus_C1_MAE','C2','C1','MAE'),
    ('C2_minus_C1_range','C2','C1','mean_range'),
    ('C2_minus_A_RMSE','C2','A','RMSE'),
)
SCENARIOS = (
    ('original_all5','raw',tuple(range(5))),
    ('clipped_all5','clipped',tuple(range(5))),
    ('original_H1H2','raw',(0,1)),
    ('clipped_H1H2','clipped',(0,1)),
)
HIDS = ['H1','H2','H3','H4','H5']

def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()

def read_arrays():
    d=pd.read_csv(ROOT/'step12_1_all_oof_predictions.csv',dtype={'audio_id':str,'speaker':str,'hypothesis_id':str})
    assert len(d)==8800 and d.audio_id.nunique()==400 and d.speaker.nunique()==20
    assert set(d.model)=={'M0','A','B1','B2','C1','C2'}
    assert d.groupby('speaker').audio_id.nunique().eq(20).all()
    assert not d.duplicated(['model','audio_id','hypothesis_id']).any()
    audios=np.array(sorted(d.audio_id.unique()),dtype=str)
    base=d.loc[d.model=='A'].set_index('audio_id').loc[audios]
    y=base.fluency.to_numpy(float)
    speakers=base.speaker.to_numpy(str)
    assert len(np.unique(speakers))==20
    pred={}
    for model in ['M0','A','B1','B2','C1','C2']:
        t=d[d.model==model]
        if model in ['M0','A']:
            t=t.set_index('audio_id').loc[audios]
            assert np.allclose(t.fluency.to_numpy(float),y)
            arr=np.repeat(t.prediction.to_numpy(float).reshape(-1,1),5,axis=1)
        else:
            arr=t.pivot(index='audio_id',columns='hypothesis_id',values='prediction').loc[audios,HIDS].to_numpy(float)
        assert arr.shape==(400,5) and np.isfinite(arr).all()
        pred[model]=arr
    assert np.isclose(np.abs(pred['C2']-y[:,None]).mean(),0.778166761,atol=1e-6)
    assert np.isclose(np.ptp(pred['C2'],axis=1).mean(),0.228505314,atol=1e-6)
    return d,audios,speakers,y,pred

def exact_signflip_p(deltas):
    """Exact two-sided paired-label-swap p from all 2^20 speaker sign assignments.
    Null requires exchangeability of two model labels within a speaker cluster.
    For RMSE, use differences in speaker-level MSE; signs of aggregated RMSE
    contrast and MSE contrast coincide, yielding the same label-swap p-value.
    """
    delta=np.asarray(deltas,dtype=float)
    assert delta.shape==(20,) and np.isfinite(delta).all()
    obs=abs(delta.sum())
    n=1<<20
    count=0
    for start in range(0,n,1<<15):
        inds=np.arange(start,min(start+(1<<15),n),dtype=np.uint32)
        # baseline all -1, flip to +1 for bits on; efficient without materializing 1Mx20
        sums=np.full(len(inds),-delta.sum(),dtype=float)
        for k,x in enumerate(delta):
            sums += ((inds>>k)&1).astype(float)*(2*x)
        count+=np.count_nonzero(np.abs(sums)>=obs-1e-11)
    return count/n, count

def pack_speaker_metrics(speaker,y,preds,processing,ids):
    cluster=sorted(set(speaker))
    assert len(cluster)==20
    idx=np.stack([np.flatnonzero(speaker==s) for s in cluster])
    assert idx.shape==(20,20)
    values={}
    for model,v in preds.items():
        a=v[:,ids]
        if processing=='clipped': a=np.clip(a,0,10)
        e=a-y[:,None]
        # First average over hypothesis indices, then recordings, then speakers;
        # each speaker has precisely 20 recordings.
        ae=np.abs(e).mean(axis=1)
        se=np.square(e).mean(axis=1)
        ran=np.ptp(a,axis=1)
        values[model]={
            'MAE':ae[idx].mean(axis=1),
            'MSE':se[idx].mean(axis=1),
            'mean_range':ran[idx].mean(axis=1)
        }
    return cluster,idx,values

def get_metric_stat(metric,left,right,stat):
    if metric=='RMSE':
        return float(np.sqrt(np.mean(stat[left]['MSE']))-np.sqrt(np.mean(stat[right]['MSE'])))
    return float(stat[left][metric].mean()-stat[right][metric].mean())

def bootstrap_dist(metric,left,right,stat,draws):
    if metric=='RMSE':
        return np.sqrt(stat[left]['MSE'][draws].mean(axis=1))-np.sqrt(stat[right]['MSE'][draws].mean(axis=1))
    return stat[left][metric][draws].mean(axis=1)-stat[right][metric][draws].mean(axis=1)

def holm_adjust(raw):
    raw=np.array(raw,dtype=float)
    ind=np.argsort(raw)
    sorted_p=raw[ind]
    adj_sorted=np.maximum.accumulate((len(raw)-np.arange(len(raw)))*sorted_p)
    adj_sorted=np.minimum(adj_sorted,1.0)
    out=np.empty(len(raw),dtype=float);out[ind]=adj_sorted
    return out

def run():
    d,audios,speakers,y,pred=read_arrays()
    rng=np.random.default_rng(SEED)
    draws=rng.integers(0,20,size=(NBOOT,20))
    rows=[];speaker_rows=[]; distributions={}
    for scen,processing,ids in SCENARIOS:
        cluster,idx,m=pack_speaker_metrics(speakers,y,pred,processing,ids)
        assert cluster==sorted(cluster)
        for contrast,left,right,metric in CONTRASTS:
            point=get_metric_stat(metric,left,right,m)
            dist=bootstrap_dist(metric,left,right,m,draws)
            testmetric='MSE' if metric=='RMSE' else metric
            delta=m[left][testmetric]-m[right][testmetric]
            pval,n_extreme=exact_signflip_p(delta)
            row={
                'scenario':scen,'contrast':contrast,'model_left':left,'model_right':right,
                'metric':metric,'point_estimate':point,
                'nominal_ci95_low':float(np.percentile(dist,2.5)),
                'nominal_ci95_high':float(np.percentile(dist,97.5)),
                'bonferroni_family4_simultaneous_ci98p75_low':float(np.percentile(dist,0.625)),
                'bonferroni_family4_simultaneous_ci98p75_high':float(np.percentile(dist,99.375)),
                'p_exact_speaker_label_swap_two_sided':pval,
                'extreme_assignments':int(n_extreme),'total_sign_assignments':1<<20,
                'bootstrap_iterations':NBOOT,'bootstrap_seed':SEED,
                'n_speakers':20,'n_audio':400,'hypotheses_per_audio':len(ids),
                'test_statistic_for_RMSE':'MSE_difference' if metric=='RMSE' else metric,
                'posthoc_family_definition':True,
                'posthoc_clipping':processing=='clipped',
            }
            rows.append(row)
            for s,l,r in zip(cluster,m[left][testmetric],m[right][testmetric]):
                speaker_rows.append({'scenario':scen,'contrast':contrast,'speaker':s,
                                    'test_statistic':testmetric,'left_speaker_mean':l,
                                    'right_speaker_mean':r,'speaker_difference':l-r})
            distributions[f'{scen}__{contrast}']=dist
    table=pd.DataFrame(rows)
    for scen in table.scenario.unique():
        mask=table.scenario.eq(scen)
        adjusted=holm_adjust(table.loc[mask,'p_exact_speaker_label_swap_two_sided'].to_numpy())
        table.loc[mask,'holm_p_adjusted_family4']=adjusted
        table.loc[mask,'holm_reject_0p05']=adjusted<0.05
        table.loc[mask,'bonferroni_CI_excludes_0']=(table.loc[mask,'bonferroni_family4_simultaneous_ci98p75_low'].gt(0) |
                                                      table.loc[mask,'bonferroni_family4_simultaneous_ci98p75_high'].lt(0))
    table['inferential_status']='EXPLORATORY_posthoc_family_not_preregistered_no_refitting_only_20_speakers'
    table.to_csv(ROOT/'step12_4_all_scenarios_exact_tests.csv',index=False,float_format='%.12g')
    table.loc[table.scenario.eq('original_all5')].to_csv(ROOT/'step12_4_primary_holm_table.csv',index=False,float_format='%.12g')
    table.loc[~table.scenario.eq('original_all5')].to_csv(ROOT/'step12_4_sensitivity_holm_table.csv',index=False,float_format='%.12g')
    pd.DataFrame(speaker_rows).to_csv(ROOT/'step12_4_speaker_paired_differences.csv',index=False,float_format='%.12g')
    np.savez_compressed(ROOT/'step12_4_bootstrap_distributions.npz',**distributions)
    method={
        'input_csv':'step12_1_all_oof_predictions.csv','input_sha256':sha(ROOT/'step12_1_all_oof_predictions.csv'),
        'pilot_audios':400,'speaker_clusters':20,'fixed_five_fold_OOF':True,
        'n_bootstrap_iterations':NBOOT,'bootstrap_seed':SEED,
        'exact_speaker_paired_label_swap_assignments':1<<20,
        'four_contrasts':list(CONTRASTS),'scenarios':[list(s[:2])+[list(s[2])] for s in SCENARIOS],
        'holm_family_size_per_scenario':4,
        'test_note':'Two-sided exact sign-flip assumes exchangeable speaker-level model labels under null; RMSE tested via MSE difference.',
        'CI_note':'Marginal 95% percentile bootstrap and conservative 98.75% Bonferroni simultaneous percentile intervals (not Holm-adjusted intervals).',
        'limitations':['Contrasts selected after initial results; non-preregistered and exploratory',
           'No multiple correction across four scenario families or across all analyses in project',
           'Bootstrap resamples only 20 speakers, does not refit scoring models',
           'Clipping and H1/H2 restriction are post-hoc sensitivity analyses',
           'No generalization to different speaker populations, ASR models or tasks']
    }
    (ROOT/'methodology.json').write_text(json.dumps(method,indent=2,ensure_ascii=False),encoding='utf-8')
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        view=table[table.scenario.eq('original_all5')].copy()
        fig,ax=plt.subplots(figsize=(9.6,4.5))
        yloc=np.arange(len(view))[::-1]
        pt=view.point_estimate.to_numpy()
        lower=view.bonferroni_family4_simultaneous_ci98p75_low.to_numpy()
        upper=view.bonferroni_family4_simultaneous_ci98p75_high.to_numpy()
        ax.errorbar(pt,yloc,xerr=[pt-lower,upper-pt],fmt='o',capsize=4)
        ax.axvline(0,color='gray',linestyle='--',linewidth=1)
        ax.set_yticks(yloc,[f'{r.contrast} ({r.metric})' for r in view.itertuples()])
        ax.set_xlabel('Paired difference (score points)')
        ax.set_title('Exploratory paired comparisons: 98.75% Bonferroni intervals')
        ax.grid(axis='x',alpha=.2)
        fig.tight_layout();fig.savefig(ROOT/'step12_4_primary_adjusted_intervals.png',dpi=175);plt.close(fig)
    except Exception as exc: print('Plot warning:',exc)
    print(table[['scenario','contrast','point_estimate','nominal_ci95_low','nominal_ci95_high',
                 'bonferroni_family4_simultaneous_ci98p75_low',
                 'bonferroni_family4_simultaneous_ci98p75_high',
                 'p_exact_speaker_label_swap_two_sided','holm_p_adjusted_family4','holm_reject_0p05']].to_string(index=False))
    return table

if __name__=='__main__':run()
