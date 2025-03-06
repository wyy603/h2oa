import os
import numpy as np
import random
import torch

@torch.jit.script
def copysign(a, b):
    # type: (float, Tensor) -> Tensor
    a = torch.tensor(a, device=b.device, dtype=torch.float).repeat(b.shape[0])
    return torch.abs(a) * torch.sign(b)
def get_euler_xyz(q):
    qx, qy, qz, qw = 0, 1, 2, 3
    # roll (x-axis rotation)
    sinr_cosp = 2.0 * (q[:, qw] * q[:, qx] + q[:, qy] * q[:, qz])
    cosr_cosp = q[:, qw] * q[:, qw] - q[:, qx] * \
                q[:, qx] - q[:, qy] * q[:, qy] + q[:, qz] * q[:, qz]
    roll = torch.atan2(sinr_cosp, cosr_cosp)

    # pitch (y-axis rotation)
    sinp = 2.0 * (q[:, qw] * q[:, qy] - q[:, qz] * q[:, qx])
    pitch = torch.where(
        torch.abs(sinp) >= 1, copysign(np.pi / 2.0, sinp), torch.asin(sinp))

    # yaw (z-axis rotation)
    siny_cosp = 2.0 * (q[:, qw] * q[:, qz] + q[:, qx] * q[:, qy])
    cosy_cosp = q[:, qw] * q[:, qw] + q[:, qx] * \
                q[:, qx] - q[:, qy] * q[:, qy] - q[:, qz] * q[:, qz]
    yaw = torch.atan2(siny_cosp, cosy_cosp)

    return torch.stack((roll, pitch, yaw), dim=-1)

def calculate_sphere(center, radius, color):
    vertices = []
    colors = []

    num_latitude_lines = 10
    num_longitude_lines = 20

    for i in range(1, num_latitude_lines):
        theta = np.pi * i / num_latitude_lines
        r = radius * np.sin(theta)
        z = center[2] + radius * np.cos(theta)

        for j in range(num_longitude_lines):
            phi1 = 2 * np.pi * j / num_longitude_lines
            phi2 = 2 * np.pi * (j + 1) / num_longitude_lines

            x1 = center[0] + r * np.cos(phi1)
            y1 = center[1] + r * np.sin(phi1)
            x2 = center[0] + r * np.cos(phi2)
            y2 = center[1] + r * np.sin(phi2)

            vertices.extend([x1, y1, z, x2, y2, z])
            colors.extend(color)

    for i in range(num_longitude_lines):
        phi = 2 * np.pi * i / num_longitude_lines

        for j in range(num_latitude_lines):
            theta1 = np.pi * j / num_latitude_lines
            theta2 = np.pi * (j + 1) / num_latitude_lines

            x1 = center[0] + radius * np.sin(theta1) * np.cos(phi)
            y1 = center[1] + radius * np.sin(theta1) * np.sin(phi)
            z1 = center[2] + radius * np.cos(theta1)

            x2 = center[0] + radius * np.sin(theta2) * np.cos(phi)
            y2 = center[1] + radius * np.sin(theta2) * np.sin(phi)
            z2 = center[2] + radius * np.cos(theta2)

            vertices.extend([x1, y1, z1, x2, y2, z2])
            colors.extend(color)

    vertices = np.array(vertices, dtype=np.float32)
    colors = np.array(colors, dtype=np.float32)

    return vertices, colors


def calculate_cube_surface(center, length, width, height):
    x, y = center[0], center[1]
    num_lines = 10

    vertices = np.array([
        [x - length / 2, y - width / 2, 0],
        [x + length / 2, y - width / 2, 0],
        [x + length / 2, y + width / 2, 0],
        [x - length / 2, y + width / 2, 0],
        [x - length / 2, y - width / 2, height],
        [x + length / 2, y - width / 2, height],
        [x + length / 2, y + width / 2, height],
        [x - length / 2, y + width / 2, height],
    ], dtype=np.float32)

    colors = [
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
        [1, 1, 0],
    ]

    def generate_lines_for_surface(v1, v2, v3, v4, num_lines):
        lines = []
        for i in range(num_lines + 1):
            alpha = i / num_lines
            p1 = (1 - alpha) * v1 + alpha * v4
            p2 = (1 - alpha) * v2 + alpha * v3
            lines.append([p1, p2])

        for i in range(num_lines + 1):
            alpha = i / num_lines
            p1 = (1 - alpha) * v1 + alpha * v2
            p2 = (1 - alpha) * v4 + alpha * v3
            lines.append([p1, p2])

        return lines

    all_lines = []
    all_colors = []

    lines = generate_lines_for_surface(vertices[0], vertices[1], vertices[5], vertices[4], num_lines)
    all_lines.extend(lines)
    all_colors.extend([[1, 0, 0]] * len(lines))

    lines = generate_lines_for_surface(vertices[1], vertices[2], vertices[6], vertices[5], num_lines)
    all_lines.extend(lines)
    all_colors.extend([[0, 1, 0]] * len(lines))

    lines = generate_lines_for_surface(vertices[0], vertices[3], vertices[7], vertices[4], num_lines)
    all_lines.extend(lines)
    all_colors.extend([[0, 0, 1]] * len(lines))

    lines = generate_lines_for_surface(vertices[3], vertices[2], vertices[6], vertices[7], num_lines)
    all_lines.extend(lines)
    all_colors.extend([[1, 1, 0]] * len(lines))

    flattened_lines = [point for line in all_lines for point in line]

    return flattened_lines, all_colors


def calculate_cylinder_surface(center, radius, height):
    center_x, center_y = center
    num_segments = 50

    lines = []
    for i in range(num_segments):
        angle1 = 2 * np.pi * i / num_segments
        angle2 = 2 * np.pi * (i + 1) / num_segments

        x1_bottom = center_x + radius * np.cos(angle1)
        y1_bottom = center_y + radius * np.sin(angle1)
        x2_bottom = center_x + radius * np.cos(angle2)
        y2_bottom = center_y + radius * np.sin(angle2)

        x1_top = x1_bottom
        y1_top = y1_bottom
        z1_top = height
        x2_top = x2_bottom
        y2_top = y2_bottom
        z2_top = height

        lines.append([[x1_bottom, y1_bottom, 0], [x2_bottom, y2_bottom, 0]])
        lines.append([[x1_top, y1_top, height], [x2_top, y2_top, height]])
        lines.append([[x1_bottom, y1_bottom, 0], [x1_top, y1_top, height]])
        lines.append([[x2_bottom, y2_bottom, 0], [x2_top, y2_top, height]])

    flattened_lines = [point for line_set in lines for point in line_set]
    colors = [[0, 0, 1]] * (len(flattened_lines) // 2)

    return flattened_lines, colors
