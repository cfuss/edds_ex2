from scipy import stats

reported_hr = [0.2403]
reproduced_hr = [1.0]

reported_ndcg = [0.1574]
reproduced_ndcg = [0.7078]

w_stat_hr, p_value_hr = stats.wilcoxon(reported_hr, reproduced_hr)
w_stat_ndcg, p_value_ndcg = stats.wilcoxon(reported_ndcg, reproduced_ndcg)

print(f"HR - Wilcoxon statistic: {w_stat_hr}, p-value: {p_value_hr}")
print(f"NDCG - Wilcoxon statistic: {w_stat_ndcg}, p-value: {p_value_ndcg}")