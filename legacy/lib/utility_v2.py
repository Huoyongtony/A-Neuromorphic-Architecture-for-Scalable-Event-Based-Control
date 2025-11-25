from __future__ import annotations
"""
Refactored Graph visualisation utility
-------------------------------------
This version restructures the original implementation for clarity and
maintainability **without changing the public API**:

* **No external behaviour changes** – the constructor and all public
  methods keep the same signatures and semantics.
* **Code reuse** – duplicated logic (e.g. animation‑script injection)
  is factored into a single static helper.
* **Clearer structure** – long methods are broken up into smaller
  private helpers; constants are grouped at the top of the file.
* **Type annotations** and **PEP‑8** compliant style throughout.
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple

import networkx as nx
import numpy as np
import webcolors
from pyvis.network import Network

__all__ = ["Graph"]

# -----------------------------------------------------------------------------
# Helper types
# -----------------------------------------------------------------------------
CentreMap = Dict[int, int]  # member_node → centre_node mapping
GraphDict = Dict[int, nx.MultiDiGraph]


class Graph:
    """Visualise synaptic relations and voltage traces using *PyVis*.

    The **API is identical** to the original implementation – only the internal
    structure and style have been improved.
    """

    # ---------------------------------------------------------------------
    # Constants
    # ---------------------------------------------------------------------
    _DEFAULT_COLOURS = [
        "red",
        "blue",
        "green",
        "orange",
        "purple",
        "brown",
        "pink",
        "gray",
        "black",
        "yellow",
    ]

    HUB_COLOUR = "#add8e6"  # LightBlue
    NEURON_COLOUR = "#cccccc"

    # ---------------------------------------------------------------------
    # Construction helpers
    # ---------------------------------------------------------------------
    @staticmethod
    def _web_to_hex(names: List[str]) -> List[str]:
        """Convert a list of *web‑safe* colour names to hex strings."""
        return [webcolors.name_to_hex(n) for n in names]

    # ---------------------------------------------------------------------
    # Public API 
    # ---------------------------------------------------------------------
    def __init__(
        self,
        syn_matrixs: np.ndarray,
        syn_taus: np.ndarray,
        syn_alphas: np.ndarray,
        shifts: np.ndarray,
        neuron_names: Dict[str, str],
        node_pos_by_name: Dict[str, Tuple[float, float]],
        node_pos_by_id: Dict[str, Tuple[float, float]],
        edge_colours_by_node_id: list[set[str], str],
    ) -> None:
        # assert input types
        assert isinstance(syn_matrixs, np.ndarray)
        assert isinstance(syn_taus, np.ndarray)
        assert isinstance(syn_alphas, np.ndarray)
        assert isinstance(shifts, np.ndarray)
        assert isinstance(neuron_names, dict)
        assert isinstance(node_pos_by_name, dict)
        assert isinstance(edge_colours_by_node_id, list)
        
        self.syn_matrixs = syn_matrixs
        self.syn_taus = syn_taus
        self.syn_alphas = syn_alphas
        self.shifts = shifts
        self.num_neuron = syn_matrixs.shape[1]
        self.neuron_names = neuron_names
        if node_pos_by_name is not None:
            self.node_pos_by_name = node_pos_by_name # dict[str, Tuple[float, float]]
            # convert neruon names in node_pos_by_name to int by looking up neuron_names and build a dict of node positions
        else:
            self.node_pos_by_name = {}
            self.node_pos_by_id = node_pos_by_id
                

        self.edge_colours_by_id = edge_colours_by_node_id

        # Visual attributes
        self._colours = self._web_to_hex(self._DEFAULT_COLOURS)
        self._colour_idx = 0

        # Results – initialised on demand
        self.net: Network | None = None
        self.per_centre: GraphDict | None = None
        self.residual_G: nx.MultiDiGraph | None = None
        self.all_clique_members: Dict[int, List[int]] | None = None
        # inside Graph.__init__
        self._graph_html: dict[str, Path] = {}      # NEW


    # ------------------------------------------------------------------
    # Internal utilities
    # ------------------------------------------------------------------
    def _next_colour(self) -> str:
        colour = self._colours[self._colour_idx]
        self._colour_idx = (self._colour_idx + 1) % len(self._colours)
        return colour

    # ..................................................................
    # Graph analysis helpers
    # ..................................................................
    @staticmethod
    def collapse_bidirectional_cliques(
        adj: np.ndarray,
    ) -> Tuple[nx.DiGraph, nx.Graph, nx.DiGraph, List[set[int]], CentreMap]:
        """Collapse every fully bidirectional clique into a centre node.

        Returns *(H, Gc, R, cliques, centre_of)* exactly like the original
        implementation.
        """
        G = nx.from_numpy_array(adj.T, create_using=nx.DiGraph)

        bidirected = nx.Graph(
            (u, v) for u, v in G.edges() if G.has_edge(v, u) and u != v
        )
        cliques = [set(c) for c in nx.find_cliques(bidirected) if len(c) > 1]

        H = G.copy()
        N = G.number_of_nodes()
        centre_of: CentreMap = {}

        for k, C in enumerate(cliques, start=1):
            centre = N + k - 1
            H.add_node(centre)
            H.remove_edges_from((u, v) for u in C for v in C if u != v)
            for u in C:
                H.add_edge(u, centre)
                H.add_edge(centre, u)
                centre_of[u] = centre

        Gc = nx.Graph()
        Gc.add_edges_from((u, centre) for u, centre in centre_of.items())

        R = H.copy()
        R.remove_nodes_from(set(centre_of.values()))

        return H, Gc, R, cliques, centre_of

    # ..................................................................
    @staticmethod
    def split_by_centre(
        full_graph: nx.MultiDiGraph, centre_of: CentreMap
    ) -> Tuple[GraphDict, nx.MultiDiGraph]:
        """Break *full_graph* into **per‑centre** sub‑graphs and a residual."""
        nodes_by_centre: Dict[int, set[int]] = {}
        for member, centre in centre_of.items():
            nodes_by_centre.setdefault(centre, {centre}).add(member)

        per_centre = {
            c: full_graph.subgraph(nodes).copy() for c, nodes in nodes_by_centre.items()
        }

        residual = full_graph.copy()
        for nodes in nodes_by_centre.values():
            residual.remove_edges_from(
                [(u, v, k) for u in nodes for v in nodes for k in residual[u].get(v, {})]
            )
        return per_centre, residual

    # ------------------------------------------------------------------
    # Private helpers for visualisation
    # ------------------------------------------------------------------
    
    def _inject_cluster(self,html_path: Path | str, clique_map: dict, colour: str) -> None:
        """Patch a *PyVis* page to collapse clique communities into clusters."""
        js = f"""
        <script>
        const cliqueMap = {json.dumps(clique_map, separators=(',', ':'))};
        function makeClusters(network) {{
            Object.entries(cliqueMap).forEach(([centre, members]) => {{
                const nodes = [Number(centre), ...members];
                network.cluster({{
                    joinCondition: n => nodes.includes(n.id),
                    clusterNodeProperties: {{
                        id: 'cluster_' + centre,
                        label: 'Centre ' + (centre - {self.num_neuron} + 1),
                        shape: 'dot',
                        size: 40,
                        color: '{colour}',
                        borderWidth: 3,
                        allowSingleNodeCluster: true
                    }}
                }});
            }});
        }}
        network.once('afterDrawing', () => makeClusters(network));
        network.on('doubleClick', params => {{
            if (params.nodes.length !== 1) return;
            const id = params.nodes[0];
            if (network.isCluster(id)) network.openCluster(id);
            else if (cliqueMap[id]) makeClusters(network);
        }});
        </script>
        """
        html_path = Path(html_path)
        html = html_path.read_text(encoding="utf-8").replace("</body>", js + "</body>", 1)
        html_path.write_text(html, encoding="utf-8")

    # ..................................................................
    @staticmethod
    def _inject_anim(html_path: Path | str, node_ids: List[int], volt_traces: np.ndarray,
                     v_min: float, v_max: float, *, fps: int = 30, autoplay: bool = False) -> None:
        template = r"""
<script>
(() => {{
    const VOLT = {volt};
    const NIDS = {nids};
    const FPS  = {fps};
    const VMIN = {vmin}, VMAX = {vmax};
    const rgb = v => {{
        const r = Math.round(255 * (v - VMIN) / (VMAX - VMIN));
        return `rgb(${{r}},0,${{255 - r}})`;
    }};
    let lastF = -1;
    function colour(f) {{
        if (f === lastF || f >= VOLT[0].length) return;
        nodes.update(NIDS.map((id, k) => ({{ id, color: {{ background: rgb(VOLT[k][f]), border: '#000' }} }})));
        lastF = f;
    }}
    window.addEventListener('message', e => {{
        if (e.data && typeof e.data.frame === 'number') colour(e.data.frame);
        if (e.data === 'reset') {{ lastF = -1; colour(0); }}
    }});
    {auto}
}})();
</script>
"""
        script = template.format(
            volt=json.dumps(volt_traces[node_ids].tolist()),
            nids=node_ids,
            fps=fps,
            vmin=v_min,
            vmax=v_max,
            auto="colour(0);" if autoplay else "",
        )
        p = Path(html_path)
        p.write_text(p.read_text(encoding="utf-8").replace("</body>", script + "</body>", 1), encoding="utf-8")

    # ------------------------------------------------------------------
    # Public visualisation methods (API preserved)
    # ------------------------------------------------------------------
    def calculate_graph(self, matrix_find_centre: List[bool]) -> dict[str, Path]:  # noqa: C901
        """Compute all graph layers and write static HTML visualisations."""
        # -- prepare containers ------------------------------------------------
        list_H: List[nx.DiGraph] = []
        list_Gc: List[nx.Graph] = []
        list_R: List[nx.DiGraph] = []
        list_centre: List[CentreMap] = []
        list_cliques: List[List[set[int]]] = []

        relation_mats: List[np.ndarray] = []
        edge_smooth, edge_arrows, edge_colours, edge_lengths, relation_names = [], [], [], [], []

        # -- iterate over relation matrices -----------------------------------
        total_nodes = self.num_neuron
        for idx, use_centre in enumerate(matrix_find_centre):
            if use_centre:
                H, Gc, R, cliques, centre_of = self.collapse_bidirectional_cliques(
                    self.syn_matrixs[idx]
                )
                list_H.append(H)
                list_Gc.append(Gc)
                list_R.append(R)
                list_centre.append(centre_of)
                list_cliques.append(cliques)
            else:
                relation_mats.append(self.syn_matrixs[idx])
                edge_smooth.append("curvedCCW")
                edge_arrows.append(True)
                edge_colours.append(self._next_colour())
                edge_lengths.append(200)
                relation_names.append(
                    f"Synapse {idx + 1}"
                )

        # -- expand hub layers --------------------------------------------------
        N = self.num_neuron
        all_clique_members: Dict[int, List[int]] = {}
        for i, Gc in enumerate(list_Gc):
            cliques = list_cliques[i]
            centre_of = list_centre[i]
            clique_members = {c: [] for c in centre_of.values()}
            for m, c in centre_of.items():
                clique_members[c].append(m)
            for c in clique_members:
                clique_members[c].append(c)
            all_clique_members.update(clique_members)

            M = len(cliques)
            Gc_mat = np.zeros((total_nodes + M, total_nodes + M), dtype=np.uint8)
            rows, cols = np.array(Gc.edges()).T
            Gc_mat[rows, cols] = 1
            relation_mats.append(Gc_mat)
            edge_smooth.append("straight")
            edge_arrows.append(False)
            colour = self._next_colour()
            edge_colours.append(colour)
            edge_lengths.append(40)
            relation_names.append(f"Hub {i}, Clique {i}")

            # residual layer ---------------------------------------------------
            R_core = nx.to_numpy_array(list_R[i], nodelist=range(N))
            R_mat = np.zeros((total_nodes + M, total_nodes + M))
            R_mat[:N, :N] = R_core.T
            relation_mats.append(R_mat)
            edge_smooth.append("curvedCCW")
            edge_arrows.append(True)
            edge_colours.append(colour)
            edge_lengths.append(200)
            relation_names.append(f"Residual {i}")

            total_nodes += M

        # -- build master MultiDiGraph ---------------------------------------
        G = nx.MultiDiGraph()
        for v in range(self.num_neuron):
            label = self.neuron_names.get(str(v), str(v))
            G.add_node(v, color=self.NEURON_COLOUR, size=10, label=label)

        # add centre nodes (after original neurons) ---------------------------
        next_id = self.num_neuron
        for clq in list_cliques:
            for _ in clq:
                G.add_node(next_id, color=self.HUB_COLOUR, size=10, label=f"Centre Node {next_id - self.num_neuron}")
                next_id += 1

        def _same_clique(u: int, v: int) -> bool:
            return any(cmap.get(u) == cmap.get(v) and cmap.get(u) is not None for cmap in list_centre)
        

        for mat_idx, M in enumerate(relation_mats):
            n = M.shape[0]
            colour = edge_colours[mat_idx]
            name = relation_names[mat_idx]
            smooth = edge_smooth[mat_idx]
            arrow = edge_arrows[mat_idx]
            length = edge_lengths[mat_idx]
            for j in range(n):
                for i in range(n):
                    w = M[i, j]
                    if w == 0:
                        continue
                    arrows_spec = (
                        {"to": {"enabled": False}}
                        if not arrow
                        else {"to": {"enabled": True, "type": "circle" if w < 0 else "arrow", "scaleFactor": 0.5}}
                    )
                    intra = _same_clique(i, j)

                    colour_temp = colour

                    for k in range(len(self.edge_colours_by_id)):
                        if (i in self.edge_colours_by_id[k][0]) and (j in self.edge_colours_by_id[k][0]):
                            colour_temp = self.edge_colours_by_id[k][1]
                            break


                    if j < self.num_neuron and mat_idx<self.syn_alphas.shape[0]:

                        G.add_edge(
                            j,
                            i,
                            title=f"{name},Tau {self.syn_taus[mat_idx,j]},Alpha {self.syn_alphas[mat_idx,j]},Shift {self.shifts[mat_idx,j]}: {w}",
                            relation=name,
                            color=colour_temp,
                            smooth={"type": smooth, "roundness": 0.2},
                            arrows=arrows_spec,
                            length=25 if intra else length,
                            physics=True,
                        )
                    else:

                        G.add_edge(
                            j,
                            i,
                            title=f"{name}",
                            relation=name,
                            color=colour_temp,
                            smooth={"type": smooth, "roundness": 0.2},
                            arrows=arrows_spec,
                            length=25 if intra else length,
                            physics=True,
                        )

        # -- build PyVis network ---------------------------------------------
        net = Network(height="700px", directed=True, notebook=False, cdn_resources="in_line")
        net.from_nx(G)
        net.force_atlas_2based()
        opts = {
            "nodes": {"shape": "dot", "size": 15, "font": {"size": 14}},
            "edges": {"arrows": {"to": {"enabled": True, "scaleFactor": 0.9}}, "smooth": {"enabled": True}},
            "physics": {"solver": "forceAtlas2Based"},
            "configure": {"enabled": True, "filter": ["physics"]},
        }
        net.set_options(json.dumps(opts))


        for node in net.nodes:
            label = node.get("label", str(node["id"]))
            if label in self.node_pos_by_name:
                x, y = self.node_pos_by_name[label]
                node.update({"x": x, "y": y, "fixed": {"x": True, "y": True}, "physics": False})

        net.save_graph("four_relations_network.html")
        self._inject_cluster("four_relations_network.html", all_clique_members, self.HUB_COLOUR)
        self._graph_html["full"] = Path("four_relations_network.html")     


        # save attributes for later dashboards ------------------------------
        self.net = net
        self.per_centre, self.residual_G = self.split_by_centre(G, list_centre[-1] if list_centre else {})
        self.all_clique_members = all_clique_members

        # build and save community and residual graphs ----------------------
        if self.per_centre:
            for centre, subG in self.per_centre.items():
                net_c = Network(height="600px", directed=True, cdn_resources="in_line")
                net_c.from_nx(subG)
                net_c.force_atlas_2based(gravity=-70, central_gravity=0.01, spring_length=30, spring_strength=0.08, damping=0.4, overlap=0)
                net_c.save_graph(f"community_{centre}.html")
                self._graph_html[f"community:{centre}"] = Path(f"community_{centre}.html")   

        if self.residual_G:
            isolates = list(nx.isolates(self.residual_G))
            self.residual_G.remove_nodes_from(isolates)
            net_res = Network(height="700px", directed=True, cdn_resources="in_line")
            net_res.from_nx(self.residual_G)
            net_res.force_atlas_2based()
            net_res.set_options(json.dumps(opts))
            net_res.save_graph("residual_edges.html")
            self._graph_html["residual"] = Path("residual_edges.html")         

        return self._graph_html


    # ------------------------------------------------------------------
    # DASHBOARDS – implemented on top of the refactored helpers
    # ------------------------------------------------------------------

    def build_dashboard(
        self,
        volt_traces: np.ndarray,
        video_file: str,
        graphs: list[str] | None = None,      # ← pick any subset here
        out_dir: str = "animations",
        play_on_load: bool = False,
        fps: int = 30,
    ) -> None:
        """
        Create a video‑plus‑graph dashboard.

        Parameters
        ----------
        graphs
            • "full"                    – the four‑relation network  
            • "residual"                – residual graph  
            • "community:<centre‑id>"   – a single community  
            • "all"                     – synonym for every available graph  
            • None (default)            – ["full"] (video + main plot only)

        returns
            The path to the graph HTML files.
        """
        if self.net is None or not self._graph_html:
            raise RuntimeError("Run calculate_graph() first.")

        if graphs is None:
            graphs = ["full"]
        elif "all" in graphs:
            graphs = list(self._graph_html)           # everything we know about

        out_dir = Path(out_dir)
        out_dir.mkdir(exist_ok=True)

        html_files: list[str] = []

        # 1) video
        video_html = self._write_video_player(video_file, out_dir, play_on_load, fps)
        html_files.append(video_html.name)

        # 2) requested graph panels
        vmin, vmax = float(volt_traces.min()), float(volt_traces.max())
        for key in graphs:
            if key not in self._graph_html:
                raise ValueError(f"Unknown graph key {key!r}. "
                                f"Known keys: {list(self._graph_html)}")

            src = self._graph_html[key]
            dst = out_dir / (src.stem + "_anim.html")        # copy + animate
            dst.write_text(src.read_text(), encoding="utf-8")

            # decide which node ids to animate
            if key == "full":
                node_ids = [n["id"] for n in self.net.nodes
                            if isinstance(n["id"], int) and n["id"] < self.num_neuron]
            elif key.startswith("community:"):
                centre = int(key.split(":")[1])
                node_ids = [n for n in self.per_centre[centre] if n < self.num_neuron]
            else:                   # residual
                node_ids = [n for n in self.residual_G if n < self.num_neuron]

            self._inject_anim(dst,
                            node_ids,
                            volt_traces,
                            vmin, vmax,
                            fps=fps,
                            autoplay=play_on_load)

            # cluster collapsing only matters for the full graph
            if key == "full":
                self._inject_cluster(dst, self.all_clique_members or {}, self.HUB_COLOUR)

            html_files.append(dst.name)

        # 3) final dashboard:
        
        dash_html = self._write_multi_panel_dashboard(html_files, out_dir)
        

        print(f"✔ Dashboard ready → {dash_html}")


    # ------------------------------------------------------------------
    # HTML writers
    # ------------------------------------------------------------------
    @staticmethod
    def _write_video_player(
        video_file: str, out_dir: Path, play_on_load: bool, fps: int
    ) -> Path:
        video_html = out_dir / "video_player.html"
        autoplay_js = "vid.play();" if play_on_load else ""
        video_html.write_text(
            f"""<!doctype html>
<html><head><meta charset='utf-8'><title>video</title></head>
<body style='margin:0'>
<video id='vid' style='width:100%;height:100%' controls>
    <source src='{video_file}' type='video/mp4'>
</video>
<script>
const vid = document.getElementById('vid');
const FPS = {fps};
window.addEventListener('message', e => {{
    if (e.data === 'start') vid.play();
    if (e.data === 'pause') vid.pause();
    if (e.data === 'reset') {{ vid.pause(); vid.currentTime = 0; parent.postMessage({{frame:0}}, '*'); }}
}});
function tick(now, meta) {{
    parent.postMessage({{frame: Math.floor(meta.mediaTime * FPS)}}, '*');
    vid.requestVideoFrameCallback(tick);
}}
vid.addEventListener('loadedmetadata', () => vid.requestVideoFrameCallback(tick));
vid.loop = false;
{autoplay_js}
</script></body></html>""",
            encoding="utf-8",
        )
        return video_html

    # ..................................................................
    @staticmethod
    def _write_two_panel_dashboard(graph_html: str, video_html: str, out_dir: Path) -> Path:
        dash_html = out_dir / "voltage_dashboard.html"
        dash_html.write_text(
            f"""<!doctype html><html><head><meta charset='utf-8'><title>Dashboard</title>
<style>
body {{ margin:0; display:flex; flex-wrap:wrap; gap:6px; font-family:sans-serif; justify-content:center; }}
.panel {{ flex:1 0 48%; height:480px; border:none; }}
.controls {{ flex:1 0 100%; text-align:center; margin:6px; }}
</style></head><body>
<div class='controls'>
    <button onclick=\"go('start')\">Play ▸</button>
    <button onclick=\"go('pause')\">Pause ❚❚</button>
    <button onclick=\"go('reset')\">Reset ⟳</button>
</div>
<iframe class='panel' src='{video_html}'></iframe>
<iframe class='panel' src='{graph_html}'></iframe>
<script>
const frames=[...document.querySelectorAll('iframe')].map(f=>f.contentWindow);
function go(cmd){{frames.forEach(f=>f.postMessage(cmd,'*'));}}
window.addEventListener('message',e=>{{if(e.data&&typeof e.data.frame==='number')frames.forEach(f=>{{if(f!==e.source)f.postMessage(e.data,'*');}});if(e.data==='reset')frames.forEach(f=>{{if(f!==e.source)f.postMessage('reset','*');}});}});
</script></body></html>""",
            encoding="utf-8",
        )
        return dash_html

    # ..................................................................
    @staticmethod
    def _write_multi_panel_dashboard(html_files: List[str], out_dir: Path) -> Path:
        dash_html = out_dir / "all_animations.html"
        iframe = "<iframe class='panel' src='{src}' width='33%' height='470' style='border:none'></iframe>"
        frames_html = "".join(iframe.format(src=s) for s in html_files)
        dash_html.write_text(
            f"""<!doctype html><html><head><meta charset='utf-8'><title>Voltage dashboard</title>
<style>
.panel {{ width:33%; height:480px; border:none; object-fit:contain; display:block; }}
body {{ margin:0; display:flex; flex-wrap:wrap; gap:6px; font-family:sans-serif; justify-content:center; }}
</style></head><body>
<div style='flex:1 0 100%;text-align:center;margin:6px'>
    <button onclick=\"go('start')\">Play ▸</button>
    <button onclick=\"go('pause')\">Pause ❚❚</button>
    <button onclick=\"go('reset')\">Reset ⟳</button>
</div>
{frames_html}
<script>
const frames=[...document.querySelectorAll('iframe')].map(f=>f.contentWindow);
function go(cmd){{frames.forEach(f=>f.postMessage(cmd,'*'));}}
window.addEventListener('message',e=>{{const d=e.data;if(d&&typeof d.frame==='number')frames.forEach(f=>{{if(f!==e.source)f.postMessage(d,'*');}});if(d==='reset')frames.forEach(f=>{{if(f!==e.source)f.postMessage('reset','*');}});}});
</script></body></html>""",
            encoding="utf-8",
        )
        return dash_html
