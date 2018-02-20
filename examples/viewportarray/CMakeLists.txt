buildExample(viewportarray viewportarray.cpp)

compile_shaders(viewportarray
    shaders/multiview.geom
    shaders/scene.frag
    shaders/scene.vert
)

link_assets(viewportarray
    models/sampleroom.dae
)
