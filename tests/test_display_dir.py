"""产物目录的展示路径：不许把盘符与 Windows 用户名下发到前端。

回归背景：README 里的界面截图曾经直接拍到界面上那行
「产物目录：E:\\<作者目录>\\outputs_web\\run_20261002_175031」——
绝对路径含盘符与用户名，而这一行恰恰最容易出现在截图、issue 和文档里。
现在服务端下发给前端的统一是「相对项目根」的路径，磁盘上仍用绝对路径。
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import web.server as server


def test_none_and_empty_return_none():
    assert server._display_dir(None) is None
    assert server._display_dir("") is None


def test_inside_project_becomes_relative_with_forward_slashes():
    """项目内的产物目录 → 相对路径，且用正斜杠（跨平台、可读）。"""
    abs_dir = os.path.join(server.PROJECT_ROOT, "outputs_web", "run_20261002_175031")
    out = server._display_dir(abs_dir)
    assert out == "outputs_web/run_20261002_175031/"
    assert "\\" not in out


def test_inside_project_is_relative_to_project_root_not_cwd():
    """相对项目根，而不是相对当前工作目录——否则换个目录启动就显示成 ../../。"""
    abs_dir = os.path.join(server.PROJECT_ROOT, "outputs_web", "run_x")
    assert server._display_dir(abs_dir).startswith("outputs_web/")


@pytest.mark.skipif(os.name != "nt", reason="跨盘符 relpath 抛错是 Windows 行为")
def test_other_drive_falls_back_to_basename():
    """跨盘符时 relpath 会抛 ValueError，必须兜住并只报最后一级。"""
    assert server._display_dir("D:\\other\\run_x") == "run_x"


def test_outside_project_falls_back_to_basename():
    """项目外的路径不给完整路径，只报最后一级——同样是为了不外泄本机目录结构。"""
    outside = os.path.join(os.path.dirname(server.PROJECT_ROOT), "somewhere", "run_x")
    assert server._display_dir(outside) == "run_x"


def test_display_path_never_contains_drive_letter_or_username():
    """总闸：真实输入域（各种位置的 run 目录）下，展示串都不许带盘符或绝对前缀。

    只取 `out_dir` 真实可能出现的形态：项目内的 run 目录、项目外同盘的 run 目录、
    以及另一个盘上的 run 目录。不含 `~`、根目录这类合成输入——它们不会成为产物目录，
    而"只留最后一级"的兜底正是靠最后一级恰好是 `run_xxx` 才安全。
    """
    cases = [
        os.path.join(server.PROJECT_ROOT, "outputs_web", "run_a"),
        os.path.join(os.path.dirname(server.PROJECT_ROOT), "outputs_web", "run_b"),
    ]
    if os.name == "nt":
        cases.append("D:\\outputs_web\\run_c")

    for raw in cases:
        out = server._display_dir(raw)
        assert out and out == out.strip(), out
        assert ":" not in out, out
        assert not out.startswith(("/", "\\")), out
        assert "run_" in out, out
