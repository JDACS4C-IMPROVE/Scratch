
# From Gwinn, Slack, 2025-07-29
def spearman(rank_actual, rank_predict):
    # Basic sum of differences (Σ d)
    sum1 = 0
    # Sum of difference squares (Σ d^2)
    sum2 = 0

    for i in range(0, all):
        drug = rank_actual.iloc[i][4]
        # print(drug)
        # print("drug ", drug)
        match = rank_predict.loc[rank_predict["run"] == drug]
        # print("match ", match.index.item())
        diff  = abs(i - match.index.item())
        sum1 += diff
        diff2 = diff ** 2
        # print("diff2 ", diff2)
        sum2 += diff2
        # print("match ", str(match), " is ", )

    spearman_all = 1 - 6*sum2/(all*(all**2 - 1))
    print(spearman_all)
