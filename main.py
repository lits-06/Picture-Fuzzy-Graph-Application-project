import networkx as nx
import heapq
import random
import time

from collections import deque


def bfs_search(G, start, end):
    queue = deque([(start, [start])])
    paths = []

    while queue:
        current_node, path = queue.popleft()

        if current_node == end:
            paths.append(path)
            continue

        for neighbor in G.neighbors(current_node):
            if neighbor not in path:
                queue.append((neighbor, path + [neighbor]))

    return paths


def dfs_search(G, start, end):
    stack = [(start, [start])]
    paths = []

    while stack:
        current_node, path = stack.pop()

        if current_node == end:
            paths.append(path)
            continue

        for neighbor in G.neighbors(current_node):
            if neighbor not in path:
                stack.append((neighbor, path + [neighbor]))

    return paths


def sorted_edge(edge):
    return tuple(sorted(edge))


# Hàm tính khoảng cách Euclid
def euclidean_distance(coord1, coord2):
    x1, y1 = coord1
    x2, y2 = coord2
    return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5


# Hàm A* để tìm tối đa 3 đường từ start đến goal
def a_star_search(G, start, goal):
    frontier = []
    heapq.heappush(frontier, (0, [start]))  # (chi phí, đường dẫn)

    paths = []  # Lưu trữ các đường dẫn tìm được

    while frontier:
        current_cost, current_path = heapq.heappop(frontier)
        current_node = current_path[-1]  # Đỉnh cuối của đường dẫn

        if current_node == goal:
            paths.append(current_path)  # Đã đến đích
            continue

        for neighbor in G.neighbors(current_node):
            if neighbor in current_path:
                continue  # Tránh đường vòng

            edge_cost = euclidean_distance(
                G.nodes[current_node]["coord"], G.nodes[neighbor]["coord"]
            )
            new_cost = current_cost + edge_cost
            new_path = current_path + [neighbor]

            heuristic = euclidean_distance(
                G.nodes[neighbor]["coord"], G.nodes[goal]["coord"]
            )
            priority = new_cost + heuristic

            heapq.heappush(frontier, (priority, new_path))

    return paths


# Hàm tính tất cả các đường dẫn giữa hai đỉnh trừ chính đường nối thẳng 2 đỉnh với nhau
def dfs(G, sorted_edge):
    start, end = sorted_edge
    stack = [(start, [start])]
    all_paths = []

    while stack:
        (current, path) = stack.pop()

        for neighbor in G.neighbors(current):
            if neighbor in path:
                continue

            if current == start and neighbor == end:
                continue

            new_path = path + [neighbor]

            if neighbor == end:
                all_paths.append(new_path)
            else:
                stack.append((neighbor, new_path))

    return all_paths


# Tính Is của từng đoạn đường
def compute_is_of_path(path, pairs):
    ic1 = 1
    ic2 = 1
    ic3 = 0

    # Duyệt qua các cặp trong đường dẫn
    for i in range(len(path) - 1):
        # Xác định cặp từ đỉnh đầu và đỉnh cuối
        edge = sorted_edge((path[i], path[i + 1]))
        pair1 = (path[i], edge)
        pair2 = (path[i + 1], edge)

        pair1_info = pairs.get(pair1)
        pair2_info = pairs.get(pair2)

        ic1 = min(ic1, pair1_info["PM"], pair2_info["PM"])
        ic2 = min(ic2, pair1_info["NM"], pair2_info["NM"])
        ic3 = max(ic3, pair1_info["nM"], pair2_info["nM"])

    return (ic1, ic2, ic3)


# Hàm xác định PFICP
def find_pficp(G, pairs):
    pficps = []

    # Duyệt qua từng cạnh trong đồ thị
    for edge in G.edges:
        edge = sorted_edge(edge)
        u, v = edge
        paths = dfs(G, edge)

        pair1 = (u, edge)
        pair2 = (v, edge)

        pair1_info = pairs.get(pair1)
        pair2_info = pairs.get(pair2)

        if len(paths) == 0:
            icn1 = max(pair1_info["PM"], pair2_info["PM"])
            icn2 = max(pair1_info["NM"], pair2_info["NM"])
            icn3 = min(pair1_info["nM"], pair2_info["nM"])

            # af = after remove
            icn1_af = pair2_info["PM"]
            icn2_af = pair2_info["NM"]
            icn3_af = pair2_info["nM"]

            if icn1_af < icn1 and icn2_af < icn2 and icn3_af > icn3:
                pficps.append(pair1)

            icn1_af = pair1_info["PM"]
            icn2_af = pair1_info["NM"]
            icn3_af = pair1_info["nM"]

            if icn1_af < icn1 and icn2_af < icn2 and icn3_af > icn3:
                pficps.append(pair2)

        else:
            ic1_path = 1
            ic2_path = 1
            ic3_path = 0

            for path in paths:
                ic1_p, ic2_p, ic3_p = compute_is_of_path(path, pairs)
                ic1_path = min(ic1_path, ic1_p)
                ic2_path = min(ic2_path, ic2_p)
                ic3_path = max(ic3_path, ic3_p)

            icn1 = max(pair1_info["PM"], pair2_info["PM"], ic1_path)
            icn2 = max(pair1_info["NM"], pair2_info["NM"], ic2_path)
            icn3 = min(pair1_info["nM"], pair2_info["nM"], ic3_path)

            # af = after remove
            icn1_af = max(pair2_info["PM"], ic1_path)
            icn2_af = max(pair2_info["NM"], ic2_path)
            icn3_af = min(pair2_info["nM"], ic3_path)

            if icn1_af < icn1 and icn2_af < icn2 and icn3_af > icn3:
                pficps.append(pair1)

            icn1_af = max(pair1_info["PM"], ic1_path)
            icn2_af = max(pair1_info["NM"], ic2_path)
            icn3_af = min(pair1_info["nM"], ic3_path)

            if icn1_af < icn1 and icn2_af < icn2 and icn3_af > icn3:
                pficps.append(pair2)

    return pficps


# Kiểm tra đoạn đường sắp đi có phải PFICP không
def check_pficp_in_path(path, pficps):
    edge = sorted_edge((path[0], path[1]))
    if (path[0], edge) in pficps:
        return True

    return False

    # for i in range(len(path) - 1):
    #     edge = sorted_edge((path[i], path[i + 1]))
    #     if (path[i], edge) in pficps:
    #         return True

    # return False


# Kiểm tra các đoạn đường, nếu đoạn nào có PFICP thì bỏ qua, nếu tất cả đoạn đường đều có PFICP
# thì lấy đoạn đường ngắn nhất
def recommed_route(paths, pficps, visited):
    for path in paths:
        if check_pficp_in_path(path, pficps):
            continue
        else:
            valid = True
            for node in path[1:]:
                if node in visited:
                    valid = False
                    break

            if valid:
                return path

    return None


def move_to_next_node(
    current_node,
    current_node_index,
    search_func,
    search_visited,
    search_path,
    search_length,
    search_time,
    search_code_runtime,
):
    if current_node != end:
        start_time = time.time()
        paths = search_func(G, current_node, end)
        path = recommed_route(paths, pficps, search_visited)

        if path:
            search_path = search_path[:current_node_index] + path

        search_code_runtime += time.time() - start_time

        next_node = search_path[current_node_index + 1]
        print(
            f"With {search_func.__name__}, currently at node: {current_node}, moving to node: {next_node}"
        )

        edge_length = euclidean_distance(
            G.nodes[current_node]["coord"], G.nodes[next_node]["coord"]
        )
        search_length += edge_length

        scale = 1
        pair = (current_node, sorted_edge((current_node, next_node)))
        if pair in pficps:
            scale = 2

        search_time += edge_length / speed * scale

        search_visited.add(next_node)
        current_node_index += 1
        current_node = next_node

    return (
        current_node,
        current_node_index,
        search_path,
        search_length,
        search_time,
        search_code_runtime,
    )


def update_pairs(pairs):
    for pair in pairs:
        PM = random.random()
        NM = random.random()
        nM = random.random()

        total = PM + NM + nM
        if total > 1:
            scale = random.random()
            scale = random.uniform(total, total + scale)
            PM /= scale
            NM /= scale
            nM /= scale

        pairs[pair]["PM"] = round(PM, 2)
        pairs[pair]["NM"] = round(NM, 2)
        pairs[pair]["nM"] = round(nM, 2)


def print_pairs_info(pairs):
    print("Pair information:")
    for key, value in pairs.items():
        total = value["PM"] + value["NM"] + value["nM"]
        total = round(total, 2)
        print("Key:", key, ", value:", value, ", Total:", total)


# Tạo đồ thị PFIG
G = nx.Graph()

G.add_nodes_from(
    [
        ("My Dinh", {"coord": (20, 40)}),
        ("Nguyen Chi Thanh", {"coord": (40, 20)}),
        ("Duong Lang", {"coord": (40, 40)}),
        ("La Thanh", {"coord": (60, 20)}),
        ("Xa Dan", {"coord": (60, 40)}),
        ("Giai Phong", {"coord": (80, 40)}),
        ("Pho Vong", {"coord": (80, 60)}),
        ("Bach Mai", {"coord": (80, 80)}),
        ("Dai Co Viet", {"coord": (60, 60)}),
        ("Thanh Nhan", {"coord": (60, 80)}),
        ("Le Duan", {"coord": (40, 80)}),
        ("Kham Thien", {"coord": (20, 60)}),
    ]
)

# Thêm các cạnh (edges) với trọng số
G.add_edges_from(
    [
        ("My Dinh", "Nguyen Chi Thanh"),
        ("Duong Lang", "My Dinh"),
        ("Duong Lang", "Nguyen Chi Thanh"),
        ("La Thanh", "Nguyen Chi Thanh"),
        ("La Thanh", "Xa Dan"),
        ("Giai Phong", "La Thanh"),
        ("Giai Phong", "Xa Dan"),
        ("Kham Thien", "My Dinh"),
        ("Dai Co Viet", "Giai Phong"),
        ("Giai Phong", "Pho Vong"),
        ("Dai Co Viet", "Kham Thien"),
        ("Dai Co Viet", "Pho Vong"),
        ("Kham Thien", "Le Duan"),
        ("Dai Co Viet", "Le Duan"),
        ("Dai Co Viet", "Thanh Nhan"),
        ("Bach Mai", "Pho Vong"),
        ("Bach Mai", "Thanh Nhan"),
        ("Le Duan", "Thanh Nhan"),
    ]
)

pairs = {
    ("My Dinh", ("My Dinh", "Nguyen Chi Thanh")): {"PM": 0.05, "NM": 0.3, "nM": 0.39},
    ("Nguyen Chi Thanh", ("My Dinh", "Nguyen Chi Thanh")): {
        "PM": 0.1,
        "NM": 0.2,
        "nM": 0.31,
    },
    ("My Dinh", ("Duong Lang", "My Dinh")): {"PM": 0.12, "NM": 0.21, "nM": 0.42},
    ("Duong Lang", ("Duong Lang", "My Dinh")): {"PM": 0.2, "NM": 0.29, "nM": 0.4},
    ("Nguyen Chi Thanh", ("Duong Lang", "Nguyen Chi Thanh")): {
        "PM": 0.2,
        "NM": 0.15,
        "nM": 0.5,
    },
    ("Duong Lang", ("Duong Lang", "Nguyen Chi Thanh")): {
        "PM": 0.2,
        "NM": 0.17,
        "nM": 0.48,
    },
    ("Nguyen Chi Thanh", ("La Thanh", "Nguyen Chi Thanh")): {
        "PM": 0.1,
        "NM": 0.06,
        "nM": 0.33,
    },
    ("La Thanh", ("La Thanh", "Nguyen Chi Thanh")): {
        "PM": 0.19,
        "NM": 0.01,
        "nM": 0.32,
    },
    ("La Thanh", ("La Thanh", "Xa Dan")): {"PM": 0.23, "NM": 0.01, "nM": 0.21},
    ("Xa Dan", ("La Thanh", "Xa Dan")): {"PM": 0.21, "NM": 0.02, "nM": 0.3},
    ("Giai Phong", ("Giai Phong", "La Thanh")): {"PM": 0.21, "NM": 0.02, "nM": 0.3},
    ("La Thanh", ("Giai Phong", "La Thanh")): {"PM": 0.21, "NM": 0.02, "nM": 0.3},
    ("Giai Phong", ("Giai Phong", "Xa Dan")): {"PM": 0.21, "NM": 0.02, "nM": 0.3},
    ("Xa Dan", ("Giai Phong", "Xa Dan")): {"PM": 0.21, "NM": 0.02, "nM": 0.3},
    ("Kham Thien", ("Kham Thien", "My Dinh")): {"PM": 0.21, "NM": 0.02, "nM": 0.3},
    ("My Dinh", ("Kham Thien", "My Dinh")): {"PM": 0.21, "NM": 0.02, "nM": 0.3},
    ("Dai Co Viet", ("Dai Co Viet", "Giai Phong")): {"PM": 0.21, "NM": 0.02, "nM": 0.3},
    ("Giai Phong", ("Dai Co Viet", "Giai Phong")): {"PM": 0.21, "NM": 0.02, "nM": 0.3},
    ("Giai Phong", ("Giai Phong", "Pho Vong")): {"PM": 0.21, "NM": 0.02, "nM": 0.3},
    ("Pho Vong", ("Giai Phong", "Pho Vong")): {"PM": 0.21, "NM": 0.02, "nM": 0.3},
    ("Dai Co Viet", ("Dai Co Viet", "Kham Thien")): {"PM": 0.21, "NM": 0.02, "nM": 0.3},
    ("Kham Thien", ("Dai Co Viet", "Kham Thien")): {"PM": 0.21, "NM": 0.02, "nM": 0.3},
    ("Dai Co Viet", ("Dai Co Viet", "Pho Vong")): {"PM": 0.21, "NM": 0.02, "nM": 0.3},
    ("Pho Vong", ("Dai Co Viet", "Pho Vong")): {"PM": 0.21, "NM": 0.02, "nM": 0.3},
    ("Kham Thien", ("Kham Thien", "Le Duan")): {"PM": 0.21, "NM": 0.02, "nM": 0.3},
    ("Le Duan", ("Kham Thien", "Le Duan")): {"PM": 0.21, "NM": 0.02, "nM": 0.3},
    ("Dai Co Viet", ("Dai Co Viet", "Le Duan")): {"PM": 0.21, "NM": 0.02, "nM": 0.3},
    ("Le Duan", ("Dai Co Viet", "Le Duan")): {"PM": 0.21, "NM": 0.02, "nM": 0.3},
    ("Dai Co Viet", ("Dai Co Viet", "Thanh Nhan")): {"PM": 0.21, "NM": 0.02, "nM": 0.3},
    ("Thanh Nhan", ("Dai Co Viet", "Thanh Nhan")): {"PM": 0.21, "NM": 0.02, "nM": 0.3},
    ("Bach Mai", ("Bach Mai", "Pho Vong")): {"PM": 0.21, "NM": 0.02, "nM": 0.3},
    ("Pho Vong", ("Bach Mai", "Pho Vong")): {"PM": 0.21, "NM": 0.02, "nM": 0.3},
    ("Bach Mai", ("Bach Mai", "Thanh Nhan")): {"PM": 0.21, "NM": 0.02, "nM": 0.3},
    ("Thanh Nhan", ("Bach Mai", "Thanh Nhan")): {"PM": 0.21, "NM": 0.02, "nM": 0.3},
    ("Le Duan", ("Le Duan", "Thanh Nhan")): {"PM": 0.21, "NM": 0.02, "nM": 0.3},
    ("Thanh Nhan", ("Le Duan", "Thanh Nhan")): {"PM": 0.21, "NM": 0.02, "nM": 0.3},
}

start = "Bach Mai"
end = "Duong Lang"
speed = 40

a_star_current_node = start
bfs_current_node = start
dfs_current_node = start

# Tổng thời gian code chạy
a_star_time = 0
bfs_time = 0
dfs_time = 0

a_star_code_runtime = 0
bfs_code_runtime = 0
dfs_code_runtime = 0

# Tổng độ dài quãng đường
a_star_length = 0
bfs_length = 0
dfs_length = 0

a_star_path = [start]
bfs_path = [start]
dfs_path = [start]

a_star_visited = set()
bfs_visited = set()
dfs_visited = set()
a_star_visited.add(start)
bfs_visited.add(start)
dfs_visited.add(start)

a_star_current_node_index = 0
bfs_current_node_index = 0
dfs_current_node_index = 0

while True:
    if (
        a_star_current_node == end
        and bfs_current_node == end
        and dfs_current_node == end
    ):
        break

    # Cập nhật giá trị PM, NM, nM của từng pair khi đến node mới (mô phỏng theo thời gian thực)
    # Hiện tại giá trị được cập nhật random sao cho 0 <= PM + NM + nM <= 1
    # Sau đó in ra toàn bộ giá trị của các pair
    update_pairs(pairs)
    print_pairs_info(pairs)
    print()

    # Tìm các pair là PFICP và in ra
    pficps = find_pficp(G, pairs)
    print("PFICPs:", pficps)

    if a_star_current_node_index == 0:
        paths = a_star_search(G, a_star_current_node, end)
        a_star_path = paths[0]
    if bfs_current_node_index == 0:
        paths = bfs_search(G, bfs_current_node, end)
        bfs_path = paths[0]
    if dfs_current_node_index == 0:
        paths = dfs_search(G, dfs_current_node, end)
        dfs_path = paths[0]

    (
        a_star_current_node,
        a_star_current_node_index,
        a_star_path,
        a_star_length,
        a_star_time,
        a_star_code_runtime,
    ) = move_to_next_node(
        a_star_current_node,
        a_star_current_node_index,
        a_star_search,
        a_star_visited,
        a_star_path,
        a_star_length,
        a_star_time,
        a_star_code_runtime,
    )

    (
        bfs_current_node,
        bfs_current_node_index,
        bfs_path,
        bfs_length,
        bfs_time,
        bfs_code_runtime,
    ) = move_to_next_node(
        bfs_current_node,
        bfs_current_node_index,
        bfs_search,
        bfs_visited,
        bfs_path,
        bfs_length,
        bfs_time,
        bfs_code_runtime,
    )

    (
        dfs_current_node,
        dfs_current_node_index,
        dfs_path,
        dfs_length,
        dfs_time,
        dfs_code_runtime,
    ) = move_to_next_node(
        dfs_current_node,
        dfs_current_node_index,
        dfs_search,
        dfs_visited,
        dfs_path,
        dfs_length,
        dfs_time,
        dfs_code_runtime,
    )

    print()

a_star_time = round(a_star_time, 2)
bfs_time = round(bfs_time, 2)
dfs_time = round(dfs_time, 2)

a_star_length = round(a_star_length, 2)
bfs_length = round(bfs_length, 2)
dfs_length = round(dfs_length, 2)

print("Speed:", speed, "km/h")
print("A* route:", a_star_path)
print(
    "Total run time:",
    a_star_time,
    "hours. Total route length:",
    a_star_length,
    "km. Total search route time:",
    a_star_code_runtime,
    "s",
)
print("BFS route:", bfs_path)
print(
    "Total run time:",
    bfs_time,
    "hours. Total route length:",
    bfs_length,
    "km. Total search route time:",
    bfs_code_runtime,
    "s",
)
print("DFS route:", dfs_path)
print(
    "Total run time:",
    dfs_time,
    "hours. Total route length:",
    dfs_length,
    "km. Total search route time:",
    dfs_code_runtime,
    "s",
)

# Trong bài toán chọn con đường tối ưu để hạn chế tắc đường, nếu có 1 đoạn là PFICP và 1 đoạn không thì nên ưu tiên
# chọn con đường nào?

# Trong bối cảnh chọn đường tối ưu để hạn chế tắc đường, việc xác định PFICP (Picture Fuzzy Incidence Cut-Pair)
# có thể đóng một vai trò quan trọng trong việc đưa ra quyết định chọn đường. Để quyết định xem nên ưu tiên chọn
# con đường có PFICP hay không, bạn cần cân nhắc các yếu tố sau:

# PFICP Là Gì Trong Bối Cảnh Chọn Đường
# PFICP là cặp đỉnh-cạnh mà khi bị loại bỏ, có thể làm cho đồ thị bị gián đoạn hoặc giảm đáng kể mức độ kết nối.
# Trong bối cảnh giao thông, đoạn đường là PFICP có thể đại diện cho "nút cổ chai" hoặc đoạn đường quan trọng
# mà nếu bị tắc, toàn bộ mạng lưới giao thông có thể bị ảnh hưởng nghiêm trọng.
