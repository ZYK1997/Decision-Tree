"""
decision tree
"""

using BenchmarkTools

import CSV

import Random

struct Leaf
    label::Int
end

struct Node
    attribute::Int
    son::Vector{Union{Leaf, Node}}
end

# dump(Node(1, [Node(2, [Leaf(1), Leaf(2)]), Leaf(2)]))

# v should be Array{Int, 2}, every attribute should be Int
getLabel(t::Leaf, v) = t.label

getLabel(t::Node, v) = getLabel(t.son[v[t.attribute]], v)

function splitData(data, attribute::Int, numOfClasses::Int)
    map(1:numOfClasses) do i
        collect(v for v in data if v[attribute] == i)
    end
end

#=
splitData([ [3, 5], [2, 3]], 1, 3)

data = [[2, 1], [3,1], [4, 1], [5, 2]]
all(v[end] == data[1][end] for v in data)
=#

function build(
    data::Vector{Vector{Int}},
    attributes::Set{Int},
    findAttribute::Function,
    labelSize::Vector{Int}
    )::Union{Leaf, Node}

    if isempty(data)
        return Leaf(0)
    elseif isempty(attributes)
        sumup = zeros(Int, labelSize[end])
        for v in data
        	sumup[v[end]] += 1
        end
        return Leaf(findmax(sumup)[2])
    elseif all(v[end] == data[1][end] for v in data)
        return Leaf(data[1][end])
    end

    attr = findAttribute(data, attributes, labelSize)
    pop!(attributes, attr)
    datas = splitData(data, attr, labelSize[attr])
    sons = map(x -> build(x, attributes, findAttribute, labelSize), datas)
    push!(attributes, attr)
    Node(attr, sons)
end

function countLabel(data, attr::Int, num::Int)
    if size(data)[1] == 0
        return zeros(num)
    end
    sumup = zeros(Int, num)
    for v in data
        sumup[v[attr]] += 1
    end
    sumup ./ size(data)[1]
end

#=
data = [[1, 2, 3], [2, 1, 1], [2, 1, 1], [2, 3, 2]]
countLabel(data, 3, 3)
=#

function findFunction_ID3(data, attributes, labelSize)::Int
    H(p) = iszero(p) ? 0.0 : -p * log2(p)
    (m, ) = size(data[1])

    H_D = sum(H, countLabel(data, m, labelSize[m])) # H(D)
    H_D_A = map(attr for attr in attributes) do attr
        datas = splitData(data, attr, labelSize[attr])
        Hs = map(datas) do D
            sum(H, countLabel(D, m, labelSize[m]))
        end
        Ps = countLabel(data, attr, labelSize[attr])
        sum(@. Hs * Ps), attr
    end
    minimum(H_D_A)[2]
end

function findFunction_C45(data, attributes, labelSize)::Int
    H(p) = iszero(p) ? 0.0 : -p * log2(p)
    (m, ) = size(data[1])

    H_D = sum(H, countLabel(data, m, labelSize[m])) # H(D)
    H_A   = map(attr for attr in attributes) do attr
        sum(H, countLabel(data, attr, labelSize[attr]))
    end
    H_D_A = map(attr for attr in attributes) do attr
        datas = splitData(data, attr, labelSize[attr])
        Hs = map(datas) do D
            sum(H, countLabel(D, m, labelSize[m]))
        end
        Ps = countLabel(data, attr, labelSize[attr])
        sum(@. Hs * Ps), attr
    end
    gain_ratio = map((x, y) -> ((H_D - y[1]) / x, y[2]), H_A, H_D_A)
    maximum(gain_ratio)[2]
end

function findFunction_CART(data, attributes, labelSize)::Int
	gini(v) = sum(p -> p * (1 - p), v)
	(m, ) = size(data[1])

	g_D = gini(countLabel(data, m, labelSize[m]))
	g_D_A = map(attr for attr in attributes) do attr
		datas = splitData(data, attr, labelSize[attr])
		Ps = countLabel(data, attr, labelSize[attr])
		Gs = map(datas) do D
			gini(countLabel(D, m, labelSize[m]))
		end
		sum(@. Gs * Ps), attr
	end
	minimum(g_D_A)[2]
end

train_set = CSV.read(pwd() * "\\lab2_data\\Car_train.csv")

function getDict(dataset)
    map(1:7) do j
        T = typeof(dataset[j][1])
        dict = Dict{T, Int}()
        numOfClasses = 0
        for w in dataset[j]
            if !haskey(dict, w)
                numOfClasses += 1
                dict[w] = numOfClasses
            end
        end
        dict
    end
end

# getDict(train_set)

function transformDataSet(dataset, dictionaries)
    ys = map(1:7) do j
        dict = dictionaries[j]
        map(dataset[j]) do w
            dict[w]
        end
    end
    (n, ) = size(ys[1])
    xs = [[ys[j][i] for j = 1:7] for i = 1:n]
end

# transformDataSet(train_set, getDict(train_set))

function k_fold_crossValidation(
    K::Int,
    dataset, dictionaries, labelSize,
    findFunction
    )
    Random.shuffle!(dataset)
    n = length(dataset)
    m = length(dictionaries)
    blockSize = floor(Int, n / K)
    map(1:K) do i
        valid_set = dataset[1 + (i - 1) * blockSize : i * blockSize]
        train_set = Vector{Vector{Int}}(undef, n - blockSize)
        train_set[1 : (i - 1) * blockSize] = dataset[1 : (i - 1) * blockSize]
        train_set[1 + (i - 1) * blockSize : n - blockSize] = dataset[1 + i * blockSize : n]

        tree = build(train_set, Set(1:(m - 1)), findFunction, labelSize)
        predict = map(v -> getLabel(tree, v), valid_set)
        numOfRight = sum(zip(predict, valid_set)) do (p, v)
            p == v[m]
        end
        numOfRight / blockSize
    end |> sum |> (x -> x / K)
end


dictionaries = getDict(train_set)
labelSize = map(d -> length(d), dictionaries)
dataset = transformDataSet(train_set, dictionaries)
k_fold_crossValidation(10, dataset, dictionaries, labelSize, findFunction_ID3)
k_fold_crossValidation(10, dataset, dictionaries, labelSize, findFunction_C45)
k_fold_crossValidation(10, dataset, dictionaries, labelSize, findFunction_CART)

#=
data = [
    [1, 1, 1, 1, 1],
    [1, 1, 1, 2, 1],
    [2, 1, 1, 1, 2],
    [3, 2, 1, 1, 2],
    [3, 3, 2, 1, 2],
    [3, 3, 2, 2, 1],
    [2, 3, 2, 2, 2],
    [1, 2, 1, 1, 1],
    [1, 3, 2, 1, 2],
    [3, 2, 2, 1, 2],
    [1, 2, 2, 2, 2],
    [2, 2, 1, 2, 2],
    [2, 1, 2, 1, 2],
    [3, 2, 1, 2, 1]]
labelSize = [3, 3, 2, 2, 2]

findFunction_CART(data, Set([1, 2, 3, 4]), labelSize)
build(data, Set([1, 2, 3, 4]), findFunction_CART, labelSize)
build(data, Set([1, 2, 3, 4]), findFunction_ID3, labelSize)
=#

#=
# 年龄 收入 学生 信用
data = [
    [1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 2, 1],
    [1, 2, 1, 1, 1, 2],
    [2, 3, 2, 1, 1, 2],
    [2, 3, 3, 2, 1, 2],
    [2, 3, 3, 2, 2, 1],
    [3, 2, 3, 2, 2, 2],
    [3, 1, 2, 1, 1, 1],
    [6, 1, 3, 2, 1, 2],
    [4, 3, 2, 2, 1, 2],
    [4, 1, 2, 2, 2, 2],
    [5, 2, 2, 1, 2, 2],
    [5, 2, 1, 2, 1, 2],
    [6, 3, 2, 1, 2, 1]]
labelSize = [6, 3, 3, 2, 2, 2]
findFunction_ID3(data, Set([1, 2, 3, 4, 5]), labelSize)
findFunction_C45(data, Set([1, 2, 3, 4, 5]), labelSize)
tree1 = build(data, Set([1, 2, 3, 4, 5]), findFunction_ID3, labelSize)
tree2 = build(data, Set([1, 2, 3, 4, 5]), findFunction_C45, labelSize)
dump(tree1)

dump(tree2)

getLabel(root, [1, 3, 1,1])
=#


#=
test_data = [
    [1, 2, 1],
    [2, 1, 2],
    [3, 2, 1]]
findFunction_ID3(test_data, Set([1, 2]), [3, 2, 2])
findFunction_C45(test_data, Set([1, 2]), [3, 2, 2])
=#
