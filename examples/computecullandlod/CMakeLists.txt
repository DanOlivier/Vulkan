buildExample(computecullandlod computecullandlod.cpp)

compile_shaders(computecullandlod
    ../../base/shaders/textoverlay.vert
    ../../base/shaders/textoverlay.frag
    ../../base/shaders/uioverlay.vert
    ../../base/shaders/uioverlay.frag

    shaders/cull.comp
    shaders/indirectdraw.frag
    shaders/indirectdraw.vert
)

copy_assets(computecullandlod
    models/suzanne_lods.dae
)

