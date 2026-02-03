# Copyright (c) 2026 EDF <caroline.cognot@agroparistech.fr

select_month = function (m::Int64, dates, Y::AbstractMatrix)
    indicesm = findall(month.(dates) .== m)
    return Y[indicesm, :]
end
season = function (dates)
    m = month.(dates)
    seasonm = [ifelse(mi ∈ [12, 1, 2], "DJF", ifelse(mi ∈ [3, 4, 5], "MAM", ifelse(mi ∈ [6, 7, 8], "JJA", "SON"))) for mi in m]
end
select_season = function (seasons, dates, Y::AbstractMatrix)
    indicesm = findall(season.(dates) .== seasons)
    return Y[indicesm, :]
end
