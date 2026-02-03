# Copyright (c) 2026 EDF <caroline.cognot@agroparistech.fr
header = "# Copyright (c) 2026 EDF <caroline.cognot@agroparistech.fr
"

for (root, _, files) in walkdir(".")
    for file in files
        endswith(file, ".jl") || continue
        path = joinpath(root, file)

        content = read(path, String)
        startswith(content, header) && continue

        open(path, "w") do io
            write(io, header * content)
        end
    end
end

# Copyright (c) 2026 EDF <caroline.cognot@agroparistech.fr
