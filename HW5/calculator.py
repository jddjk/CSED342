from collections import deque

grid = [
    "%%%%%%%%%%%%%%%%%%%%",
    "%......%G  G%......%",
    "%.%%...%%  %%...%%.%",
    "%.%o.%........%.o%.%",
    "%.%%.%.%%%%%%.%.%%.%",
    "%........P.........%",
    "%%%%%%%%%%%%%%%%%%%%"
]

directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

def is_within_bounds(x, y):
    return 0 <= x < len(grid[0]) and 0 <= y < len(grid)

def is_passable(x, y):
    return grid[y][x] != '%'

def bfs(start):
    queue = deque([start])
    distances = {start: 0}
    while queue:
        current = queue.popleft()
        cx, cy = current
        for dx, dy in directions:
            nx, ny = cx + dx, cy + dy
            if is_within_bounds(nx, ny) and is_passable(nx, ny) and (nx, ny) not in distances:
                queue.append((nx, ny))
                distances[(nx, ny)] = distances[(cx, cy)] + 1
    return distances

all_distances = {}
for y in range(len(grid)):
    for x in range(len(grid[0])):
        if is_passable(x, y):
            all_distances[(x, y)] = bfs((x, y))

# Python 딕셔너리 형태로 출력
print("precomputed_distances = {")
for start, distances in all_distances.items():
    print(f"    {start}: {distances},")
print("}")