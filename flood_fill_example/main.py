from dataclasses import dataclass
from typing import List, Optional, TypedDict, Tuple
from pathlib import Path
from enum import Enum

DIRECTION_DICT = {
    "up": (0, -1),
    "down": (0, 1),
    "left": (-1, 0),
    "right": (1, 0),
}


class EdgeType(Enum):
    CORNER = "X"
    PARAMETER = "-"
    CONTAINED = "*"


@dataclass
class Point:
    x_coord: int
    y_coord: int

    def area_to(self, other: "Point") -> int:
        return (abs(self.x_coord - other.x_coord) + 1) * (
            abs(self.y_coord - other.y_coord) + 1
        )

    def __hash__(self):
        return hash((self.x_coord, self.y_coord))


@dataclass
class Node:
    this_node: Point
    connect_points: Optional["Node"]
    head: bool
    tail: bool

    def __hash__(self):
        return hash(self.this_node)


class ReturnDict(TypedDict):
    nodes: Node
    max_row: int
    max_col: int


def draw_grid(max_row: int, max_col: int) -> List[List[str]]:
    # We are not given a grid, so we guess by doubling the max col seen and max row seen
    return [[" " for _ in range((max_col + 1) * 2)] for _ in range((max_row + 1) * 2)]


def populate_grid(raw_grid: List[List[str]]):
    for row in raw_grid:
        first_index = next(
            (i for i, item in enumerate(row) if item in ["X", "-"]), None
        )
        last_index = next(
            (i for i in range(len(row) - 1, -1, -1) if row[i] in ["X", "-"]), None
        )

        if first_index is not None and last_index is not None:
            for i in range(first_index + 1, last_index):
                if row[i] in ["X", "-"]:
                    continue
                row[i] = EdgeType.CONTAINED.value


def build_graph(data: List[str]) -> ReturnDict | None:
    last_node = None
    head_node = None
    max_row, max_col = 0, 0

    for row in data:
        x_y_list = [*map(int, row.split(","))]
        current_node_x, current_node_y = x_y_list

        max_row = max(max_row, current_node_y)
        max_col = max(max_col, current_node_x)

        current_node = Point(current_node_x, current_node_y)

        if last_node is None:
            head_node = Node(current_node, None, True, False)
            last_node = head_node
        else:
            this_node = Node(current_node, None, False, False)
            last_node.connect_points = this_node
            last_node = this_node

    if last_node is None:
        return None

    last_node.tail = True
    last_node.connect_points = head_node

    return_dict: ReturnDict = {
        "nodes": head_node,
        "max_row": max_row,
        "max_col": max_col,
    }

    return return_dict


def draw_edges(
    nodes: Node, raw_grid: List[List[str]], delimiter: EdgeType = EdgeType.CORNER
) -> List[List[str]]:
    curr = nodes
    seen = set()
    while True:
        x, y = curr.this_node.x_coord, curr.this_node.y_coord
        raw_grid[y][x] = delimiter.value
        curr = curr.connect_points

        if curr in seen:
            break

        seen.add(curr)

    return raw_grid


def draw_until_end(
    grid: List[List[str]],
    start: Tuple[int, int],
    end: Tuple[int, int],
    direction: Tuple[int, int],
    value: EdgeType,
):
    x, y = start
    while True:
        x, y = x + direction[0], y + direction[1]
        if (x, y) == end:
            break
        grid[y][x] = value.value


def draw_parameters(
    nodes: Node, raw_grid: List[List[str]], value: EdgeType = EdgeType.PARAMETER
):
    curr = nodes
    direction = None

    while True:
        x, y = curr.this_node.x_coord, curr.this_node.y_coord
        next_node: Node = curr.connect_points
        next_x, next_y = next_node.this_node.x_coord, next_node.this_node.y_coord
        diff_x, diff_y = next_x - x, next_y - y

        if diff_x == 0:
            # We need to move right or left
            if diff_y > 0:
                direction = DIRECTION_DICT["down"]
            else:
                direction = DIRECTION_DICT["up"]
        else:
            # We need to move up or down
            if diff_x > 0:
                direction = DIRECTION_DICT["right"]
            else:
                direction = DIRECTION_DICT["left"]

        draw_until_end(raw_grid, (x, y), (next_x, next_y), direction, value)
        curr = next_node
        if curr.head:
            break


def build_flood_file(file_path: Path) -> List[List[str]]:
    data = file_path.read_text().splitlines()
    nodes = build_graph(data)
    assert nodes, "Could not build nodes"
    raw_grid = draw_grid(nodes["max_row"], nodes["max_col"])
    draw_edges(nodes["nodes"], raw_grid)
    draw_parameters(nodes["nodes"], raw_grid)
    populate_grid(raw_grid)
    return raw_grid


def display_grid(grid: List[List[str]]):
    for row in grid:
        print(row)


def main():
    data_path = Path("mock_data/connected_edges_txt.txt")
    raw_grid = build_flood_file(data_path)
    display_grid(raw_grid)


if __name__ == "__main__":
    main()
