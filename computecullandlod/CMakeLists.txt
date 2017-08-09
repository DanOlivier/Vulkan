buildExample(computecullandlod computecullandlod.cpp)

compile_shaders(computecullandlod
    shaders/cull.comp
    shaders/indirectdraw.frag
    shaders/indirectdraw.vert
)

copy_assets(computecullandlod
    models/suzanne_lods.dae
)

