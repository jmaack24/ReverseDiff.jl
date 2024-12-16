
# Extensions #
#------------#

function value(input::AbstractArray{T}) where {T<:TrackedReal}
    vals = similar(input, valtype(eltype(input)))
    for i in eachindex(input)
        vals[i] = value(input[i])
    end
    return vals
end

function value(input::Tuple)
    return map(value, input)
end

function value!(input::AbstractArray{T}, vals::AbstractArray) where {T<:TrackedReal}
    for i in eachindex(input, vals)
        value!(input[i], vals[i])
    end
    return
end

# function value!(input::AbstractArray{T}, vals::AbstractArray{S}) where {T<:TrackedReal, S<:TrackedReal}
#     for i in eachindex(input, vals)
#         value!(input[i], value(vals[i]))
#     end
#     return
# end

function value!(tup::Tuple, vals::Tuple)
    for i in eachindex(tup, vals)
        value!(input[i], vals[i])
    end
    return
end

function deriv(vals::AbstractArray{T}) where {T <: TrackedReal}
    dv = similar(vals, derivtype(eltype(vals)))
    for i in eachindex(IndexLinear(), vals)
        dv[i] = deriv(vals[i])
    end
    return dv
end

function deriv(vals::Tuple)
    return map(deriv, vals)
end

function deriv!(input::AbstractArray{T}, vals::AbstractArray) where {T<:TrackedReal}
    for i in eachindex(input, vals)
        deriv!(input[i], vals[i])
    end
    return
end

function increment_deriv!(to_incr::Tuple, to_add::Tuple)
    for i in eachindex(to_incr, to_add)
        increment_deriv!(to_incr[i], to_add[i])
    end
    return
end

# function value(avtr::Vector{T}) where T<:TrackedReal
#     return broadcast(value, avtr)
# end

# function value(avtr::Matrix{T}) where {T<:TrackedReal}
#     return broadcast(value, avtr)
# end

# function value!(avtr::Vector{T}, av::AbstractVector) where T<:TrackedReal
#     broadcast(value!, avtr, av)
#     return
# end

# function deriv(avtr::Vector{T}) where T<:TrackedReal
#     return broadcast(deriv, avtr)
# end

# function deriv(avtr::Matrix{T}) where {T<:TrackedReal}
#     return broadcast(value, avtr)
# end

# function deriv!(avtr::Vector{T}, av::AbstractVector) where {T<:TrackedReal}
#     broadcast(deriv!, avtr, av)
#     return
# end

# HDTape #
#--------#

struct HDTape{I,O} <: AbstractInternalTape
    internal_tape::InstructionTape
    input::I
    output::O
end

# function HDTape(
#     tp::InstructionTape,
#     input::I,
#     output::O,
# ) where {I, O}
#     return HDTape{I,O}(tp,
#                        input,
#                        output)
# end

@inline internal(hdtp::HDTape) = hdtp.internal_tape
@inline input(hdtp::HDTape) = hdtp.input
@inline output(hdtp::HDTape) = hdtp.output

@inline forward_pass!(hdtp::HDTape) = forward_pass!(internal(hdtp))
@inline reverse_pass!(hdtp::HDTape) = reverse_pass!(internal(hdtp))

# Base.pop!(ctp::HDTape) = pop!(internal(ctp))
# Base.push!(ctp::HDTape, item) = push!(internal(ctp), item)
# Base.empty!(ctp::HDTape) =  empty!(internal(ctp))
# Base.getindex(ctp::HDTape, idx) = getindex(internal(ctp), idx)
# Base.iterate(ctp::HDTape) = iterate(internal(ctp))
# Base.iterate(ctp::HDTape, idx) = iterate(internal(ctp), idx)
# Base.length(ctp::HDTape) = length(internal(ctp))
# Base.lastindex(ctp::HDTape) = lastindex(internal(ctp))

# function record!(hdtp::HDTape, ::Type{IT}, args...) where IT
#     return record!(internal(hdtp), IT, args...)
# end

# function load_tape(hdtp::HDTape)
#     return
# end

# function write_tape(hdtp::HDTape, tp::InstructionTape)
#     return
# end

# function forward_pass!(hdtp::HDTape)
#     tp = load_tape(hdtp)
#     fpv = forward_pass!(tp)
#     write_tape(hdtp, tp)
#     # update_tape(hdtp, tp)
#     return fpv
# end

# function reverse_pass!(hdtp::HDTape)
#     tp = load_tape(hdtp)
#     rpv = reverse_pass!(tp)
#     write_tape(hdtp, tp)
#     return rpv
# end


# CompressedTape #
#----------------#

struct CompressedTape <: AbstractInternalTape
    internal_tape::InstructionTape
    tracked_map::Dict{UInt, TrackedType}
end

@inline internal(ctp::CompressedTape) = ctp.internal_tape
@inline trap(ctp::CompressedTape) = ctp.tracked_map

@inline function CompressedTape()
    # return CompressedTape(InstructionTape(), InstructionTape())
    # return CompressedTape(InstructionTape(), Dict{UInt, TrackedType}(), Any[])
    return CompressedTape(InstructionTape(), Dict{UInt,TrackedType}())
end

Base.pop!(ctp::CompressedTape) = pop!(internal(ctp))
Base.push!(ctp::CompressedTape, item) = push!(internal(ctp), item)
function Base.empty!(ctp::CompressedTape)
    empty_tape!(ctp)
    empty_trap!(ctp)
    return
end
Base.getindex(ctp::CompressedTape, idx) = getindex(internal(ctp), idx)
Base.iterate(ctp::CompressedTape) = iterate(internal(ctp))
Base.iterate(ctp::CompressedTape, idx) = iterate(internal(ctp), idx)
Base.length(ctp::CompressedTape) = length(internal(ctp))
Base.lastindex(ctp::CompressedTape) = lastindex(internal(ctp))

empty_tape!(ctp::CompressedTape) = empty!(internal(ctp))
empty_trap!(ctp::CompressedTape) = empty!(trap(ctp))


function find_tracked(ctp::CompressedTape, cin::Tuple)
    return map(i -> find_tracked(ctp, i), cin)
end

function find_tracked(ctp::CompressedTape, cin::AbstractArray)
    return map(i -> find_tracked(ctp, i), cin)
end

function find_tracked(ctp::CompressedTape, cin::TrackedType)
    oid = objectid(cin)
    rcin = trap(ctp)[oid]
    return rcin
end

function reduce_precision(full::T) where T
    return convert(reduce_precision_type(T), full)
end

function reduce_precision(array::AbstractArray)
    T = reduce_precision_type(eltype(array))
    rar = similar(array, T)
    rar .= array
    return rar
end

# function reduce_precision(a)
#     # Default to no-op
#     return a
# end

function reduce_precision_type(::Type{Float64})
    return Float32
end

function reduce_precision_type(::Type{Int64})
    return Int32
end

function reduce_precision(
    full::TrackedReal{V, D, O},
    ctp::CompressedTape,
) where {V,D,O}

    v = reduce_precision(value(full))
    d = reduce_precision(deriv(full))
    o = hasorigin(full) ? compressed_tracked(ctp, full.origin) : nothing

    rpt = TrackedReal{typeof(v), typeof(d), typeof(o)}(
        v,
        d,
        ctp,
        full.index,
        o,
    )

    return rpt

end

function reduce_precision(
    full::TrackedArray{V,D,N,VA,DA},
    ctp::CompressedTape,
) where {V,D,N,VA,DA}

    v = reduce_precision(value(full))
    d = reduce_precision(deriv(full))

    rar = TrackedArray(v, d, ctp)

    return rar

end

function compressed_tracked(::CompressedTape, v::Real)
    return reduce_precision(v)
end

function compressed_tracked(::CompressedTape, v::AbstractArray)
    return reduce_precision(v)
end

function compressed_tracked(ctp::CompressedTape, tr::TrackedReal{V,D,O}) where {V,D,O}

    oid = objectid(tr)

    if oid in keys(trap(ctp))

        reduced_prec = ctp.tracked_map[oid]

    else

        reduced_prec = reduce_precision(tr, ctp)
        ctp.tracked_map[oid] = reduced_prec

    end

    return reduced_prec

end

function compressed_tracked(ctp::CompressedTape, ta::TrackedArray)

    oid = objectid(ta)

    if oid in keys(trap(ctp))

        rpar = ctp.tracked_map[oid]

    else

        rpar = reduce_precision(ta, ctp)
        ctp.tracked_map[oid] = rpar

    end

    return rpar

end

function compressed_tracked(ctp::CompressedTape, tp::Tuple)
    return map(i -> compressed_tracked(ctp, i), tp)
end

function compress_instruction(scalar::ScalarInstruction, ctp::CompressedTape)

    compressed = ScalarInstruction(
        scalar.func,
        compressed_tracked(ctp, scalar.input),
        compressed_tracked(ctp, scalar.output),
        scalar.cache,
    )

    return compressed

end

function compress_instruction(special::SpecialInstruction, ctp::CompressedTape)

    compressed = SpecialInstruction(
        special.func,
        compressed_tracked(ctp, special.input),
        compressed_tracked(ctp, special.output),
        special.cache,
    )
    return compressed
end

function record!(ctp::CompressedTape, ::Type{IT}, args...) where IT

    oi = IT(args...)
    ci = compress_instruction(oi, ctp)
    push!(ctp, ci)

    return

end

function forward_pass!(ctp::CompressedTape)
    # println("FORWARD!")
    return forward_pass!(internal(ctp))
end

function reverse_pass!(ctp::CompressedTape)
    # println("REVERSE!")
    return reverse_pass!(internal(ctp))
end

# CompressedInstruction #
#-----------------------#

struct CompressedInstruction{F,II,TI,IO,TO} <: AbstractInstruction
    func::F
    tape::AbstractInternalTape # All instructions to be compressed
    instruction_input::II
    tape_input::TI
    instruction_output::IO
    tape_output::TO
    # disable default outer constructor
    function CompressedInstruction{F,II,TI,IO,TO}(func, tape, input, cinput, output, coutput) where {F,II,TI,IO,TO}
        # println("Making compressed instruction")
        return new{F,II,TI,IO,TO}(func, tape, input, cinput, output, coutput)
    end
end

@inline function _CompressedInstruction(
    func::F,
    tape::T,
    input::II,
    cinput::TI,
    output::IO,
    coutput::TO
) where {F,T,II,TI,IO,TO}
    return CompressedInstruction{F,II,TI,IO,TO}(func, tape, input, cinput, output, coutput)
end

function CompressedInstruction(func, tape, input, cinput, output, coutput)
    return _CompressedInstruction(func, tape, capture(input), cinput, capture(output), coutput)
end

function Base.show(io::IO, ci::CompressedInstruction, pad="")
    my_func = string(ci.func)
    println(io, pad, "Compressed($my_func)")
    println(io, pad, "  input:    ", compactrepr(ci.instruction_input))
    println(io, pad, "  tape_in:  ", compactrepr(ci.tape_input))
    println(io, pad, "  output:   ", compactrepr(ci.instruction_output))
    print(  io, pad, "  tape_out: ", compactrepr(ci.tape_output))
    # print(  io, pad, "  cache:  ", compactrepr(ci.))
    return
end

@noinline reverse_exec!(ci::CompressedInstruction) = compressed_reverse_exec!(ci)

@noinline function compressed_reverse_exec!(ci)

    pull_value!(ci.instruction_output)
    pull_deriv!(ci.instruction_output)
    value!(ci.tape_output, value(ci.instruction_output))
    deriv!(ci.tape_output, deriv(ci.instruction_output))
    unseed!(ci.tape_input)

    reverse_pass!(ci.tape)

    # deriv!(ci.instruction_input, deriv(ci.tape_input))
    increment_deriv!(ci.instruction_input, deriv(ci.tape_input))
    unseed!(ci.instruction_output)

    return

end

@noinline forward_exec!(ci::CompressedInstruction) = compressed_forward_exec!(ci)

@noinline function compressed_forward_exec!(ci)

    pull_value!(ci.instruction_input)
    value!(ci.tape_input, value(ci.instruction_input))

    forward_pass!(ci.tape)

    value!(ci.instruction_output, value(ci.tape_output))

    return

end

# Disk Instruction #
#------------------#

mutable struct DiskInstruction{F,I,O} <: AbstractInstruction
    func::F
    fname::String
    fio::Union{Nothing, IOStream}
    # jldio::Union{Nothing, JLD2.JLDFile{JLD2.MmapIO}}
    # tape::HDTape
    instruction_input::I
    instruction_output::O
    # disable default outer constructor
    function DiskInstruction{F,I,O}(func, input, output) where {F,I,O}
        # println("Making disk instruction")
        di = new{F,I,O}(func, "", nothing, input, output)
        # finalizer(destruct, di)
        return di
    end
end

function destruct(di::DiskInstruction)
    close(di.fio)
    di.fio = nothing
    # di.jldio = nothing
    return
end

function _DiskInstruction(
    func::F,
    tape::T,
    input::II,
    output::IO,
) where {F,T,II,IO}

    di = DiskInstruction{F,II,IO}(func, input, output)

    (my_path, my_io) = mktemp(pwd(); cleanup=true)
    di.fname = my_path
    # di.fio = my_io
    close(my_io)

    # @show my_path
    # @show length(internal(tape))

    write_tape(di, tape)

    return di

end

function DiskInstruction(func, tape, input, output)
    return _DiskInstruction(func, tape, capture(input), capture(output))
end

function Base.show(io::IO, di::DiskInstruction, pad="")
    my_func = string(di.func)
    println(io, pad, "Disk($my_func)")
    println(io, pad, "  input:    ", compactrepr(di.instruction_input))
    println(io, pad, "  output:   ", compactrepr(di.instruction_output))
    print(  io, pad, "  file:     ", string(di.fname))
    return
end

function load_tape(di::DiskInstruction)
    hdtp = Serialization.deserialize(di.fname)
    return hdtp
end

function write_tape(di::DiskInstruction, tp::AbstractInternalTape)
    Serialization.serialize(di.fname, tp)
    return
end

@noinline reverse_exec!(di::DiskInstruction) = disk_reverse_exec!(di)

@noinline function disk_reverse_exec!(di)

    hdtp = load_tape(di)
    tin = input(hdtp)
    tout = output(hdtp)

    pull_value!(di.instruction_output)
    pull_deriv!(di.instruction_output)
    value!(tout, value(di.instruction_output))
    deriv!(tout, deriv(di.instruction_output))
    push_deriv!(tout)
    unseed!(tin)

    # @show di.instruction_output
    # @show tout
    # @show deriv(di.instruction_output)
    # @show deriv(tout)

    reverse_pass!(hdtp)

    # @show tin
    # @show deriv(tin)

    increment_deriv!(di.instruction_input, deriv(tin))
    unseed!(di.instruction_output)

    # @show di.instruction_input
    # @show deriv(di.instruction_input)

    return

end

@noinline forward_exec!(di::DiskInstruction) = disk_forward_exec!(di)

@noinline function disk_forward_exec!(di)

    hdtp = load_tape(di)
    tin = input(hdtp)
    tout = output(hdtp)

    pull_value!(di.instruction_input)
    value!(tin, value(di.instruction_input))

    # @show tin
    # @show di.instruction_input

    forward_pass!(hdtp)

    value!(di.instruction_output, value(tout))

    # @show tout
    # @show di.instruction_input

    write_tape(di, hdtp)

    return

end

# Compression Functions #
#-----------------------#

function compress(f, x::Real, args...)
    return f(x, args...)
end

function compress(f, x::AbstractArray, args...)
    return f(x, args...)
end

function compress(f, x::TrackedType, args...)

    tp = tape(x)
    ctp = CompressedTape()

    cin = track(value(x), ctp)
    cout = f(cin, args...)
    y = track(value(cout), tp)

    cin = find_tracked(ctp, cin)
    cout = find_tracked(ctp, cout)

    empty_trap!(ctp)

    record!(tp, CompressedInstruction, f, ctp, x, cin, y, cout)

    return y

end

# function original_type(x::TrackedType)
#     return valtype(x)
# end

# function original_type(::AbstractArray{T}) where T<:TrackedType
#     return valtype(T)
# end

# function original_type(x::Tuple)
#     return map(original_type, x)
# end

function single_precision(f, x::Real, args...)
    return f(convert(Float32, x), args...)
end

function single_precision(f, x::AbstractArray, args...)
    return f(convert.(Float32, x), args...)
end

function _single_precision(f, x, args...)

    # println("Building single CompressedInstruction...")

    tp = tape(x)
    ctp = InstructionTape()

    cin = tracked_copy(x, ctp, Float32)
    cout = apply(f, cin, args...)
    y = tracked_copy(cout, tp, Float64)

    record!(tp, CompressedInstruction, f, ctp, x, cin, y, cout)

    return y

end

single_precision(f, x::TrackedType, args...) = _single_precision(f, x, args...)
single_precision(f, x::AbstractArray{T}, args...) where {T<:TrackedType} = _single_precision(f, x, args...)
single_precision(f, x::TrackedType, y::TrackedType, args...) = _single_precision(f, (x, y), args...)
single_precision(f, x::AbstractArray{S}, y::AbstractArray{T}, args...) where {S<:TrackedType,T<:TrackedType} = _single_precision(f, (x, y), args...)

function tracked_copy(x::TrackedReal, tp, ::Type{T}=valtype(x)) where T
    return track(convert(T, value(x)), tp)
end

function tracked_copy(x::TrackedArray, tp, ::Type{T}=valtype(x)) where T
    return track(convert.(T, value(x)), tp)
end

function tracked_copy(x::AbstractArray{T}, tp, ::Type{R}=valtype(eltype(x))) where {T <: TrackedReal, R}
    my_et = TrackedReal{valtype(T), derivtype(T), Nothing}
    x_cp = similar(x, my_et)
    for i in eachindex(IndexLinear(), x)
        xi = x[i]
        x_cp[i] = TrackedReal(value(xi), deriv(xi), tp)
    end
    return x_cp
end

function tracked_copy(x::Tuple, tp, ::Type{T}) where T
    list = Any[]
    for i in eachindex(x)
        push!(list, tracked_copy(x[i], tp, T))
    end
    return Tuple(list)
end

function tracked_copy(x::Tuple, tp)
    list = Any[]
    for i in eachindex(x)
        push!(list, tracked_copy(x[i], tp))
    end
    return Tuple(list)
end

function tracked_copy(x::Tuple, tp, types::Tuple)
    list = Any[]
    for i in eachindex(x, types)
        push!(list, tracked_copy(x[i], tp, types[i]))
    end
    return Tuple(list)
end

function apply(f, in, args...)
    return f(in, args...)
end

function apply(f, in::Tuple, args...)
    return f(in..., args...)
end

function tape_to_disk(f, x, args...)
    return f(x, args...)
end

function _tape_to_disk(f, x, args...)

    # println("Building DiskInstruction...")

    tp = tape(first(x))
    ctp = InstructionTape()

    @assert tp !== NULL_TAPE

    cin = tracked_copy(x, ctp)

    cout = apply(f, cin, args...)
    y = tracked_copy(cout, tp)

    # @show ctp

    hdtp = HDTape(ctp, cin, cout)

    record!(tp, DiskInstruction, f, hdtp, x, y)

    return y

end

# function ChainRulesCore.rrule(::typeof(_tape_to_disk), f, x, p)

#     tp = tape(first(x))
#     ctp = InstructionTape()

#     cin = tracked_copy(x, ctp)
#     cout = apply(f, cin, p)
#     y = tracked_copy(cout, tp)

#     return

# end

tape_to_disk(f, x::TrackedType, args...) = _tape_to_disk(f, x, args...)
tape_to_disk(f, x::AbstractArray{T}, args...) where {T <: TrackedReal} = _tape_to_disk(f, x, args...)
tape_to_disk(f, x::TrackedType, y::TrackedType, args...) = _tape_to_disk(f, (x,y), args...)
# tape_to_disk(f, x::AbstractArray{T}, y::AbstractArray{T}, args...) where {T <: TrackedReal} = _tape_to_disk(f, (x,y), args...)
# tape_to_disk(f, x::AbstractArray{T}, y::TrackedArray, args...) where {T<:TrackedReal} = _tape_to_disk(f, (x, y), args...)
# tape_to_disk(f, x::TrackedArray, y::AbstractArray{T}, args...) where {T<:TrackedReal} = _tape_to_disk(f, (x, y), args...)

function do_compression(expr, f_to_apply)
    args = map(MacroTools.splitarg, expr.args)
    @assert(length(args) > 1)
    # @show args
    f = args[1][1]
    x = args[2][1]
    # y = length(args) > 2 ? args[3:end] : nothing

    retex = :(ReverseDiff.$(f_to_apply)($f, $x))

    if length(args) > 2
        for arg in args[3:end]
            push!(retex.args, arg[1])
        end
    end

    return retex

end

"""
Macro for compressing the tape (including inputs and outputs when approapriate) of a function call
"""
macro compress(ex)
    return esc(do_compression(ex, :compress))
end


macro single(ex)
    return esc(do_compression(ex, :single_precision))
end


macro disk(ex)
    return esc(do_compression(ex, :tape_to_disk))
end
