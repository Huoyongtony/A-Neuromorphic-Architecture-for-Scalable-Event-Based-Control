import json
import numpy as np
import networkx as nx
from pyvis.network import Network
import webcolors
from pathlib import Path


class Graph:

    def __init__(self, syn_matrixs, syn_taus, syn_alphas, shifts, neuron_names):
        self.syn_matrixs = syn_matrixs
        self.syn_taus = syn_taus
        self.syn_alphas = syn_alphas
        self.shifts = shifts
        self.num_neuron = syn_matrixs.shape[1]
        self.neuron_names = neuron_names
        self.color_index = 0

        self.colors = [webcolors.name_to_hex("red"),
                       webcolors.name_to_hex("blue"),
                       webcolors.name_to_hex("green"),
                       webcolors.name_to_hex("orange"),
                       webcolors.name_to_hex("purple"),
                       webcolors.name_to_hex("brown"),
                       webcolors.name_to_hex("pink"),
                       webcolors.name_to_hex("gray"),
                       webcolors.name_to_hex("black"),
                       webcolors.name_to_hex("yellow")]

        self.net=None
        self.per_centre=None
        self.residual_G=None
        self.relation_names=None
        self.edge_smooth=None
        self.edge_arrows=None
        self.edge_colours=None
        self.list_of_matrix_H=None
        self.list_of_matrix_Gc=None
        self.list_of_matrix_R=None
        self.list_of_relation_matrix=None
        self.list_of_centre=None
        self.list_of_cliques=None


    def collapse_bidirectional_cliques(self,adj):
        """
        Collapse every bidirectional (all-to-all) clique into a single centre node
        and return three graphs:
            H  – full digraph with centres,
            Gc – undirected graph of centre–member edges,
            R  – residual digraph without any centre edges.
        Also returns the list of cliques it found.
        """
        # --- 1. build the starting DiGraph ------------------------------------
        A = np.asarray(adj)
        G = nx.from_numpy_array(A.T, create_using=nx.DiGraph)          # column → row :contentReference[oaicite:0]{index=0}

        # --- 2. identify bidirectionally complete sub-graphs -------------------
        bidirected = nx.Graph(
            (u, v) for u, v in G.edges() if G.has_edge(v, u) and u != v
        )                                                               # undirected view :contentReference[oaicite:1]{index=1}
        cliques = [set(c) for c in nx.find_cliques(bidirected)          # Bron–Kerbosch :contentReference[oaicite:2]{index=2}
                if len(c) > 1]

        # --- 3. collapse each clique into its own centre ----------------------
        H          = G.copy()
        N          = G.number_of_nodes()
        centre_of  = {}                       # member → centre mapping

        for k, C in enumerate(cliques, start=1):
            centre = N + k - 1
            H.add_node(centre)                                                # add super-node :contentReference[oaicite:3]{index=3}
            H.remove_edges_from((u, v) for u in C for v in C if u != v)        # drop internal arcs
            for u in C:                                                       # wire member ↔ centre
                H.add_edge(u, centre)
                H.add_edge(centre, u)
                centre_of[u] = centre                                          # remember mapping

        # --- 4. build the undirected hub view --------------------------------
        Gc = nx.Graph()
        Gc.add_edges_from((u, centre) for u, centre in centre_of.items())      # batch add :contentReference[oaicite:4]{index=4}

        # --- 5. build the residual digraph -----------------------------------
        R = H.copy()
        R.remove_nodes_from(set(centre_of.values()))                           # strip centres :contentReference[oaicite:5]{index=5}


        return H, Gc, R, cliques, centre_of



    def split_by_centre(self,full_graph: nx.MultiDiGraph,
                        centre_of: dict[int, int]) -> tuple[dict[int, nx.MultiDiGraph],
                                                            nx.MultiDiGraph]:
        """
        Parameters
        ----------
        full_graph : nx.MultiDiGraph
            The graph that already contains ALL relations (Hub, Residual, B, C, D).
        centre_of : dict {member_node: centre_node}
            Mapping built while collapsing cliques.

        Returns
        -------
        per_centre : dict {centre_node: nx.MultiDiGraph}
            Induced sub-graph on (centre + its members) for every centre.
        residual   : nx.MultiDiGraph
            A copy of `full_graph` minus all edges that lie entirely inside any one
            centre’s community.
        """
        # --- 1. invert mapping → nodes grouped per centre ---------------
        nodes_by_centre = {}
        for member, centre in centre_of.items():
            nodes_by_centre.setdefault(centre, {centre}).add(member)

        # --- 2. build SubGraph views for every centre -------------------
        per_centre = {c: full_graph.subgraph(nodes).copy()   # copy = writable :contentReference[oaicite:2]{index=2}
                    for c, nodes in nodes_by_centre.items()}

        # --- 3. build residual graph -----------------------------------
        residual = full_graph.copy()
        # remove ALL edges whose two ends are in the same community
        for c, nodes in nodes_by_centre.items():
            # fast bulk removal via node pair comprehension
            residual.remove_edges_from([(u, v, k)
                                        for u in nodes for v in nodes
                                        for k in residual[u].get(v, {})])
        return per_centre, residual

    def select_color(self):
        self.color_index += 1
        if self.color_index >= len(self.colors):
            self.color_index = 0
        return self.colors[self.color_index]

    def calculate_graph(self,matrix_find_centre):

        list_of_matrix_H=[]
        list_of_matrix_Gc=[]
        list_of_matrix_R=[]
        list_of_relation_matrix=[]
        list_of_centre=[]
        list_of_cliques=[]
        tot=self.num_neuron

        edge_smooth    = []
        edge_arrows    = []
        edge_colours   = []

        relation_names = []

        N=self.num_neuron
        
        # matrix_find_centre is a 1D list with number of element equal to syn_matrixs.shape[0]
        for i in range(len(matrix_find_centre)):
            if matrix_find_centre[i]==True:
                H, Gc, R, cliques, centre_of=self.collapse_bidirectional_cliques(self.syn_matrixs[i,:,:])
                list_of_matrix_H.append(H)
                list_of_matrix_Gc.append(Gc)
                list_of_matrix_R.append(R)
                list_of_centre.append(centre_of)
                list_of_cliques.append(cliques)
            else:
                list_of_relation_matrix.append(self.syn_matrixs[i,:,:])
                edge_smooth.append('curvedCCW')
                edge_arrows.append(True)
                edge_colours.append(self.select_color())
                relation_names.append(f"Synapse {i+1},Tau {self.syn_taus[i]},Alpha {self.syn_alphas[i]},Shift {self.shifts[i]}")
                

        HUB_COLOUR    = "#add8e6"      # LightBlue  :contentReference[oaicite:0]{index=0}
        NEURON_COLOUR = "#cccccc" 

        G=nx.MultiDiGraph()
        for v in range(self.num_neuron):
            if f"{v+1}" in self.neuron_names:
                G.add_node(v, color=NEURON_COLOUR, size=10,label=self.neuron_names[f"{v+1}"])
            else:
                G.add_node(v, color=NEURON_COLOUR, size=10,label=f"{v+1}")

        for i in range(len(list_of_matrix_Gc)):
            M=len(list_of_cliques[i])
            for j in range(M):
                G.add_node(tot+j, color=HUB_COLOUR, size=10,label=f"Centre Node {tot+j+1}")
    
            Gc_mat_ = np.zeros((N+M, N+M), dtype=np.uint8)
            Gc_mat = np.zeros((tot+M, tot+M), dtype=np.uint8)
            rows, cols = np.array(list_of_matrix_Gc[i].edges()).T     # shape (2, |E|)  ⇢ vectorised indices
            Gc_mat_[rows, cols] = 1

            Gc_mat[:N,:N] = Gc_mat_[:N,:N]
            Gc_mat[tot:,tot:] = Gc_mat_[N:,N:]
            Gc_mat[:N,tot:] = Gc_mat_[:N,N:]
            Gc_mat[tot:,:N] = Gc_mat_[N:,:N]

            list_of_relation_matrix.append(Gc_mat)
            edge_smooth.append('straight')
            edge_arrows.append(False)
            color = self.select_color()
            edge_colours.append(color)
            relation_names.append(f"Hub {i+1}, Clique {i+1}")

            tot+=M

            R_core   = nx.to_numpy_array(list_of_matrix_R[i], nodelist=range(N))          # i→j orientation  :contentReference[oaicite:0]{index=0}
            R_mat    = np.zeros((tot, tot))
            R_mat[:N, :N] = R_core.T
            list_of_relation_matrix.append(R_mat)
            edge_smooth.append('curvedCCW')
            edge_arrows.append(True)
            edge_colours.append(color)
            relation_names.append(f"Residual {i+1}")

       
        for rel_idx, M in enumerate(list_of_relation_matrix):
            colour, name, smooth, arrow_flag = (edge_colours[rel_idx],
                                                relation_names[rel_idx],
                                                edge_smooth[rel_idx],
                                                edge_arrows[rel_idx])
            n=M.shape[0]
            for j in range(n):          # source column
                for i in range(n):      # target row
                    w = M[i, j]
                    if w == 0:
                        continue        # no edge ↦ skip

                    # ---- 1A.  choose arrow type ---------------------------------
                    if not arrow_flag:                   # Hub layer = undirected
                        arrows_spec = {"to": {"enabled": False}}

                    else:                                # Directed layers
                        arrow_type = "circle" if w < 0 else "arrow"
                        arrows_spec = {"to": {"enabled": True,
                                            "type": arrow_type,          # <-- key line
                                            "scaleFactor": 0.5}}

                    # ---- 1B.  add the edge --------------------------------------
                    G.add_edge(
                        j, i,
                        title=f"{name}: {w}",
                        relation=name,
                        color=colour,
                        smooth={"type": smooth, "roundness": 0.2},
                        arrows=arrows_spec
                    )


        # --- 3.  BUILD THE PyVis VISUAL  -------------------------------------
        net = Network(
            height="700px",
            directed=True,
            notebook=False,
            cdn_resources="in_line"
        )

        net.from_nx(G)

        # stable layout
        net.force_atlas_2based()

        # visual options
        opts = {
            "nodes":  {"shape": "dot", "size": 15, "font": {"size": 14}},
            "edges":  {"arrows": {"to": {"enabled": True, "scaleFactor": 0.9}},
                    "smooth": {"enabled": True}},
            "physics":{"solver": "forceAtlas2Based"},

            "configure": {
                "enabled": True,
                "filter": ["physics"]        # list or the string "physics"
            }
        }

        # *** IMPORTANT: set options BEFORE adding show_buttons ***
        net.set_options(json.dumps(opts))

        # write the HTML file
        net.save_graph("four_relations_network.html")

        self.net=net

        self.per_centre, self.residual_G = self.split_by_centre(G, centre_of)

        # visual options
        opts = {
            "nodes":  {"shape": "dot", "size": 15, "font": {"size": 14}},
            "edges":  {"arrows": {"to": {"enabled": True, "scaleFactor": 0.9}},
                    "smooth": {"enabled": True}},
            "physics":{"solver": "forceAtlas2Based",
                       },

            "configure": {
                "enabled": True,
                "filter": ["physics"]        # list or the string "physics"
            }
        }

        for centre, subG in self.per_centre.items():
            net_c = Network(height="600px", directed=True, cdn_resources="in_line")
            net_c.from_nx(subG)
            net_c.force_atlas_2based(gravity=-70,
                        central_gravity=0.01,
                        spring_length=30,
                        spring_strength=0.08,
                        damping=0.4,
                        overlap=0)
            # net_c.set_options(json.dumps(opts))          # reuse your global options
            net_c.save_graph(f"community_{centre}.html")
          
            
        isolates = list(nx.isolates(self.residual_G))          # nodes with total degree 0  :contentReference[oaicite:0]{index=0}
        self.residual_G.remove_nodes_from(isolates)  
        net_res = Network(height="700px", directed=True, cdn_resources="in_line")
        net_res.from_nx(self.residual_G)
        net_res.force_atlas_2based()
        net_res.set_options(json.dumps(opts))
        net_res.save_graph("residual_edges.html")
    

  
    def build_voltage_dashboard(
            self,
            volt_traces,                # ndarray (N, T)
            video_file,                 # "run_capture.mp4"
            out_dir="animations",
            play_on_load=False,
            fps=30):
        """
        1. Creates animated HTMLs for every graph in self.per_centre and
        for self.residual_G, colouring biological-neuron nodes by voltage.
        2. Builds a dashboard that embeds all graphs + the MP4.
        Play/Pause/Reset control everything via postMessage, so it works
        even when opened from file://
        """
        

        Path(out_dir).mkdir(exist_ok=True)
        N, T = volt_traces.shape
        v_min, v_max = float(volt_traces.min()), float(volt_traces.max())

        # ----- tiny helper --------------------------------------------------------
        def _json_opts():
            # pyvis.set_options() needs a JSON *string*
            return json.dumps(self.net.options if isinstance(self.net.options, dict)
                            else json.loads(self.net.options))

        # ----- inject JS into every graph page ------------------------------------
        def _inject_anim(html_path, node_ids, volt_traces, v_min, v_max, auto_js="",fps=30):
            """
            Post-process *html_path* to add a script that animates node colours.
            Uses str.format() so you don’t need to count braces manually.
            """
            TEMPLATE = r"""
                <script>
                (() => {{
                    const VOLT = {VOLT_JSON};                  /* ndarray (N,T)              */
                    const NIDS = {NODE_IDS};                   /* biological-neuron IDs      */
                    const FPS  = {FPS};                        /* not used, but useful       */
                    const VMIN = {V_MIN}, VMAX = {V_MAX};

                    /* blue → red ramp ---------------------------------------------------- */
                    const rgb = v => {{
                        const r = Math.round(255 * (v - VMIN) / (VMAX - VMIN));
                        return `rgb(${{r}},0,${{255 - r}})`;
                    }};

                    let lastFrame = -1;                        /* last frame we have drawn   */

                    /* recolour the graph ------------------------------------------------- */
                    function colourFrame(f) {{                 /* only if something changed  */
                        if (f === lastFrame || f >= VOLT[0].length) return;

                        const upd = NIDS.map((id, k) => ({{
                            id,
                            color: {{ background: rgb(VOLT[k][f]), border: '#000' }}
                        }}));
                        nodes.update(upd);                     /* vis-network batch update   */
                        lastFrame = f;
                    }}

                    /* listen for broadcasts coming from the video iframe ----------------- */
                    window.addEventListener('message', e => {{
                        if (e.data && typeof e.data.frame === 'number') {{
                            colourFrame(e.data.frame);         /* sync to that video frame   */
                        }}
                        if (e.data === 'reset') {{             /* dashboard reset button     */
                            lastFrame = -1;
                            colourFrame(0);
                        }}
                    }});
                }})();
                </script>
                """





            script = TEMPLATE.format(
                VOLT_JSON=json.dumps(volt_traces[node_ids].tolist()),
                V_MIN=v_min,
                V_MAX=v_max,
                NODE_IDS=node_ids,
                AUTO_JS=auto_js,
                FPS=fps          # "startAnim();" if you want autoplay
            )

            # insert once, just before </body>
            with open(html_path, "r", encoding="utf-8") as f:
                html = f.read()
            html = html.replace("</body>", script + "\n</body>", 1)
            with open(html_path, "w", encoding="utf-8") as f:
                f.write(html)

        # ----- build animated HTMLs ----------------------------------------------
        video_html = Path(out_dir) / "video_player.html"
        video_html.write_text(f"""<!doctype html>
        <html><head><meta charset="utf-8"><title>video</title></head>
        <body style="margin:0">
        <video id="vid" style="width:100%;height:100%" controls>
            <source src="{video_file}" type="video/mp4">
        </video>

        <script>
        const vid  = document.getElementById("vid");
        const FPS  = 30;                      /* ← your recording fps                */

        /* respond to Play / Pause / Reset coming from the dashboard -------------- */
       
        window.addEventListener("message", e => {{
            if (e.data === "start")  vid.play();
            if (e.data === "pause")  vid.pause();

            if (e.data === "reset"){{          /* ← patched block */
                vid.pause();
                vid.currentTime = 0;          /* seek */
                parent.postMessage({{frame: 0}}, "*");     /* tell graphs          */
                vid.requestVideoFrameCallback(tick);     /* RE-ARM the callback! */
            }}
        }});



        /* when *our* video advances, tell every graph which frame it is on -------- */
        function tick(){{
            const f = Math.floor(vid.currentTime * FPS);   /* ← one line – always works */
            parent.postMessage({{frame: f}}, "*");           /* broadcast to dashboard   */
            vid.requestVideoFrameCallback(tick);           /* re-arm callback          */
        }}


        /* kick-off once the metadata is ready ------------------------------------ */
        vid.addEventListener("loadedmetadata", () => {{
            vid.requestVideoFrameCallback(tick);
        }});

        vid.loop = false;                   /* stop on the last frame, no rewind     */
        </script>

        </body></html>""", encoding="utf-8")


        html_files = [video_html.name]

        # 1. community sub-graphs
        for c, g in self.per_centre.items():
            net = Network("600px", directed=True, cdn_resources="in_line")
            net.from_nx(g)
            net.force_atlas_2based(gravity=-70, central_gravity=0.01,
                                spring_length=30, spring_strength=0.08,
                                damping=0.4, overlap=0)
            #net.set_options(_json_opts())
            f = Path(out_dir)/f"community_{c}_anim.html"
            net.save_graph(str(f))
            _inject_anim(
                    html_path=f,
                    node_ids=[n for n in g if n < N],
                    volt_traces=volt_traces,
                    v_min=v_min,
                    v_max=v_max,
                    auto_js="startAnim();" if play_on_load else "",
                    fps=fps
                ) # biological neurons only
            html_files.append(f.name)

        # 2. residual graph
        net_r = Network("700px", directed=True, cdn_resources="in_line")
        net_r.from_nx(self.residual_G)
        net_r.force_atlas_2based()
        net_r.set_options(_json_opts())
        f_r = Path(out_dir)/"residual_edges_anim.html"
        net_r.save_graph(str(f_r))
        _inject_anim(
                    html_path=f_r,
                    node_ids=[n for n in self.residual_G if n < N],
                    volt_traces=volt_traces,
                    v_min=v_min,
                    v_max=v_max,
                    auto_js="startAnim();" if play_on_load else "",
                    fps=fps
                )
        html_files.append(f_r.name)

        # ----- master dashboard ---------------------------------------------------
        iframe = '<iframe src="{src}" width="33%" height="470" style="border:none"></iframe>'
        frames_html = "".join(iframe.format(src=s) for s in html_files)

        # -----------------------------------------------------------------
        # 1. build the iframe markup first → NO backslash in the f-string
        # -----------------------------------------------------------------
        frames_html_panel = frames_html.replace(
            '<iframe ',
            '<iframe class="panel" '
        )

        # -----------------------------------------------------------------
        # 2. assemble the dashboard safely (only two placeholders!)
        # -----------------------------------------------------------------
        dash = f"""<!doctype html><html><head>
        <meta charset="utf-8"><title>Voltage dashboard</title>

        <style>
        .panel {{
            width : 33%;
            height: 480px;
            border:none;
            display:block;
            object-fit:contain;
        }}
        body {{
            margin:0;
            display:flex;
            flex-wrap:wrap;
            gap:6px;
            font-family:sans-serif;
            justify-content:center;
        }}
        </style>
        </head><body>

        <div style="flex:1 0 100%;text-align:center;margin:6px">
        <button onclick="go('start')">Play ▸</button>
        <button onclick="go('pause')">Pause ❚❚</button>
        <button onclick="go('reset')">Reset ⟳</button>
        </div>

        {frames_html_panel}

        <script>
        const frames = [...document.querySelectorAll('iframe')].map(f => f.contentWindow);

        /* buttons --------------------------------------------------------------- */
        function go(cmd){{
            frames.forEach(f => f.postMessage(cmd, '*'));
        }}

        /* relay frame-update objects coming from the video iframe --------------- */
        window.addEventListener("message", e => {{
            const data = e.data;

            if (data && typeof data.frame === "number"){{      /* 0,1,2,… INCLUDING 0 */
                frames.forEach(f => {{ if (f !== e.source) f.postMessage(data, "*"); }});
            }}

            if (data === "reset"){{                            /* plain string        */
                frames.forEach(f => {{ if (f !== e.source) f.postMessage("reset", "*"); }});
            }}
        }});

        </script>



        </body></html>"""



        with open(Path(out_dir)/"all_animations.html", "w", encoding="utf-8") as fh:
            fh.write(dash)

        print(f"✔ Dashboard ready in {out_dir}/all_animations.html")

