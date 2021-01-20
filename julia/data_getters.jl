module DataGetters

    HEAD = 0
    BODY = 1
    DIR = 2
    FOOD = 3
    SCORE = 4
    HAS_FOOD = 5
    ALIVE = 6
    HUNGER = 7
    POSITIVE_RESPONSE = 100.0
    NEGATIVE_RESPONSE = -20.0
    NEUTRAL_RESPONSE = 1.0
    BORDER_THRESHOLD = 10.0
    HUNGER_FACTOR = 0.001
    TILE_SIZE = 20
    TERRAIN_DIMS = (30, 30)
    HEIGHT = TERRAIN_DIMS[1] * TILE_SIZE
    WIDTH = TERRAIN_DIMS[2] * TILE_SIZE
    function data_simple(state)
        xs = []
        fx, fy = state[FOOD]
        dir = state[DIR]
        hx, hy = state[HEAD]
        fx, fy = state[FOOD]
        xs = vcat(xs, [hx, hy])
        xs = vcat(xs, [fx, fy])
        goal = NEUTRAL_RESPONSE 
        dist = abs(fx - hx) + abs(fy - hy)
        append!(xs, dist)
        goal -= HUNGER_FACTOR * state[HUNGER]
        if !state[HAS_FOOD]
            goal = POSITIVE_RESPONSE
        end
        
        dist_wall_x = min(abs(hx - WIDTH), hx)
        dist_wall_y = min(abs(hy - HEIGHT), hy)
        dist_wall_food_x = min(abs(fx - WIDTH), fx)
        dist_wall_food_y = min(abs(fy - HEIGHT), fy)
        xs = vcat(xs, [dist_wall_x, dist_wall_y])
        xs = vcat(xs, [dist_wall_food_x, dist_wall_food_y])
        min_wall_dist = min(dist_wall_x, dist_wall_y)

        append!(xs, state[DIR])
        append!(xs, size(state[BODY]))
        append!(xs, state[HUNGER])
        return (xs, [goal])
    end

    function data_sensory(state, border_look_ahead=5)
        dir = state[DIR]
        fx, fy = state[FOOD]
        hx, hy = state[HEAD]
        body = state[BODY]
        data = []
        for alpha in dir-90:90:dir+90
            sensed = [0.0, 0.0, 0.0]
            for j in 0:(max(WIDTH, HEIGHT)/TILE_SIZE)
                mul = 1.0
                x = hx + j * TILE_SIZE * cos(deg2rad(alpha))
                y = hy + j * TILE_SIZE * cos(deg2rad(alpha))
                if isapprox(x, fx) && isapprox(y, fy)
                    sensed[1] = 1 / (1 + j + rand(Float32) * 2)
                end
                if j < border_look_ahead && !((0 <= x < WIDTH) && (0 <= y < HEIGHT))
                    sensed[2] = 1 / (1 + j + rand(Float32) * 2)
                end
                if j > 0
                    for part in body
                        if isapprox(x, part[1]) && isapprox(y, part[2])
                            sensed[3] = 1 / (1 + j + rand(Float32) * 2)
                        end
                    end
                end
            end
            append!(data, sensed)

        end
        #Computing how many neighbouring tiles from the head are occupied
        #by the snake's body.
        closeness = 0
        for tx in (hx - TILE_SIZE):TILE_SIZE:(hx + TILE_SIZE)
            for ty in (hy - TILE_SIZE):TILE_SIZE:(hy + TILE_SIZE)
                for (px, py) in body[2:end] #2 to not take the head into account
                    if isapprox(px, tx) && isapprox(py, ty)
                        closeness += 1
                    end
                end
            end
        end
        xs, y = data_simple(state)
        return (vcat(data, xs), y)

    end
    export data_sensory
    export data_simple
end