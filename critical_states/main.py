import pyjulia

julia_module = pyjulia.Pyjulia("./crit_states.jl")
julia_module.julia_interpreter = "C:\Users\shlok\AppData\Local\Microsoft\WindowsApps\julia.exe"

julia_module