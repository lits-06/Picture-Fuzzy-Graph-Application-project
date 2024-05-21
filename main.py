import networkx as nx
import heapq
import random


def sorted_edge(edge):
    return tuple(sorted(edge))


# Hàm tính khoảng cách Euclid
def euclidean_distance(coord1, coord2):
    x1, y1 = coord1
    x2, y2 = coord2
    return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5


# Hàm A* để tìm tối đa 3 đường từ start đến goal
def a_star_search(G, start, goal, max_paths=3):
    frontier = []
    heapq.heappush(frontier, (0, [start]))  # (chi phí, đường dẫn)

    paths = []  # Lưu trữ các đường dẫn tìm được

    while frontier and len(paths) < max_paths:
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


# Nếu đường có PFICP thì trả về True, còn không thì False
def check_pficp_in_path(path, pficps):
    for i in range(len(path) - 1):
        edge = sorted_edge((path[i], path[i + 1]))
        if (path[i], edge) in pficps:
            return True

    return False


# Kiểm tra các đoạn đường, nếu đoạn nào có PFICP thì bỏ qua, nếu tất cả đoạn đường đều có PFICP
# thì lấy đoạn đường ngắn nhất
def recommed_route(paths, pficps):
    for path in paths:
        if check_pficp_in_path(path, pficps):
            continue
        else:
            return path

    return paths[0]


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
        ("India", {"coord": (20, 40)}),
        ("UAE", {"coord": (40, 20)}),
        ("Russia", {"coord": (40, 40)}),
        ("Nicaragua", {"coord": (60, 20)}),
        ("Mexico", {"coord": (60, 40)}),
    ]
)

# Thêm các cạnh (edges) với trọng số
G.add_edges_from(
    [
        ("India", "UAE"),
        ("India", "Russia"),
        ("Russia", "UAE"),
        ("UAE", "Nicaragua"),
        ("Nicaragua", "Mexico"),
    ]
)

pairs = {
    ("India", ("India", "UAE")): {"PM": 0.05, "NM": 0.3, "nM": 0.39},
    ("UAE", ("India", "UAE")): {"PM": 0.1, "NM": 0.2, "nM": 0.31},
    ("India", ("India", "Russia")): {"PM": 0.12, "NM": 0.21, "nM": 0.42},
    ("Russia", ("India", "Russia")): {"PM": 0.2, "NM": 0.29, "nM": 0.4},
    ("UAE", ("Russia", "UAE")): {"PM": 0.2, "NM": 0.15, "nM": 0.5},
    ("Russia", ("Russia", "UAE")): {"PM": 0.2, "NM": 0.17, "nM": 0.48},
    ("UAE", ("Nicaragua", "UAE")): {"PM": 0.1, "NM": 0.06, "nM": 0.33},
    ("Nicaragua", ("Nicaragua", "UAE")): {"PM": 0.19, "NM": 0.01, "nM": 0.32},
    ("Nicaragua", ("Mexico", "Nicaragua")): {"PM": 0.23, "NM": 0.01, "nM": 0.21},
    ("Mexico", ("Mexico", "Nicaragua")): {"PM": 0.21, "NM": 0.02, "nM": 0.3},
}

start = "India"
end = "Mexico"
current_node = start

while current_node != end:
    # Cập nhật giá trị PM, NM, nM của từng pair khi đến node mới (mô phỏng theo thời gian thực)
    # Sau đó in ra toàn bộ giá trị của các pair
    update_pairs(pairs)
    print_pairs_info(pairs)
    print()

    # Tìm các pair là PFICP và in ra
    pficps = find_pficp(G, pairs)
    print("PFICPs:", pficps)

    # Thuật toán tìm đường A* tìm đường từ node hiện tại đến đích, lấy 3 đường ngắn nhất
    paths = a_star_search(G, current_node, end)

    # Xét 3 đường tìm được từ A* ở trên, nếu đường nào có PFICP (quan trọng) thì bỏ qua, xét
    # đường tiếp theo, nếu cả 3 đường đều có PFICP thì lấy đường có độ dài ngắn nhất
    recommended_path = recommed_route(paths, pficps)

    # Đi đến node tiếp theo
    next_node = recommended_path[1]
    print()
    print(f"Currently at node: {current_node}, moving to node: {next_node}")

    # Chuyển sang node tiếp theo
    current_node = next_node

# Trong bài toán chọn con đường tối ưu để hạn chế tắc đường, nếu có 1 đoạn là PFICP và 1 đoạn không thì nên ưu tiên
# chọn con đường nào?

# Trong bối cảnh chọn đường tối ưu để hạn chế tắc đường, việc xác định PFICP (Picture Fuzzy Incidence Cut-Pair)
# có thể đóng một vai trò quan trọng trong việc đưa ra quyết định chọn đường. Để quyết định xem nên ưu tiên chọn
# con đường có PFICP hay không, bạn cần cân nhắc các yếu tố sau:

# PFICP Là Gì Trong Bối Cảnh Chọn Đường
# PFICP là cặp đỉnh-cạnh mà khi bị loại bỏ, có thể làm cho đồ thị bị gián đoạn hoặc giảm đáng kể mức độ kết nối.
# Trong bối cảnh giao thông, đoạn đường là PFICP có thể đại diện cho "nút cổ chai" hoặc đoạn đường quan trọng
# mà nếu bị tắc, toàn bộ mạng lưới giao thông có thể bị ảnh hưởng nghiêm trọng.
