module Helpers
    function retroAffect(xs, ys, reward, horizon=10)

        cuttingIndex = max(1, length(ys) - horizon)
        correctedHorizon = min(length(ys), horizon)
        for (i, y) in enumerate(ys[cuttingIndex:end])
            ys[cuttingIndex + (i - 1)][1] = y[1] + (reward / max((correctedHorizon - (i - 1)), 1))
        end
        
        return (xs, ys)
    end
    export retroAffect
end
