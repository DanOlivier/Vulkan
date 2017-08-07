/*
* Vulkan Example - Scene rendering
*
* Copyright (C) 2016 by Sascha Willems - www.saschawillems.de
*
* This code is licensed under the MIT license (MIT) (http://opensource.org/licenses/MIT)
*
* Summary:
* Renders a scene made of multiple parts with different materials and textures.
*
* The example loads a scene made up of multiple parts into one vertex and index buffer to only
* have one (big) memory allocation. In Vulkan it's advised to keep number of memory allocations
* down and try to allocate large blocks of memory at once instead of having many small allocations.
*
* Every part has a separate material and multiple descriptor sets (set = x layout qualifier in GLSL)
* are used to bind a uniform buffer with global matrices and the part's material's sampler at once.
*
* To demonstrate another way of passing data the example also uses push constants for passing
* material properties.
*
* Note that this example is just one way of rendering a scene made up of multiple parts in Vulkan.
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <vector>

#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <assimp/Importer.hpp> 
#include <assimp/scene.h>     
#include <assimp/postprocess.h>
#include <assimp/cimport.h>

#include <vulkan/vulkan.hpp>
#include "vulkanexamplebase.h"
#include "VulkanTexture.hpp"
#include "VulkanDevice.hpp"
#include "VulkanBuffer.hpp"

#define VERTEX_BUFFER_BIND_ID 0
#define ENABLE_VALIDATION false

// Vertex layout used in this example
struct Vertex {
	glm::vec3 pos;
	glm::vec3 normal;
	glm::vec2 uv;
	glm::vec3 color;
};

// Scene related structs

// Shader properites for a material
// Will be passed to the shaders using push constant
struct SceneMaterialProperties
{
	glm::vec4 ambient;
	glm::vec4 diffuse;
	glm::vec4 specular;
	float opacity;
};

// Stores info on the materials used in the scene
struct SceneMaterial
{
	std::string name;
	// Material properties
	SceneMaterialProperties properties;
	// The example only uses a diffuse channel
	vks::Texture2D diffuse;
	// The material's descriptor contains the material descriptors
	vk::DescriptorSet descriptorSet;
	// Pointer to the pipeline used by this material
	vk::Pipeline *pipeline;
};

// Stores per-mesh Vulkan resources
struct ScenePart
{
	// Index of first index in the scene buffer
	uint32_t indexBase;
	uint32_t indexCount;

	// Pointer to the material used by this mesh
	SceneMaterial *material;
};

// Class for loading the scene and generating all Vulkan resources
class Scene
{
private:
	vks::VulkanDevice *vulkanDevice;
	vk::Queue queue;

	vk::DescriptorPool descriptorPool;

	// We will be using separate descriptor sets (and bindings)
	// for material and scene related uniforms
	struct
	{
		vk::DescriptorSetLayout material;
		vk::DescriptorSetLayout scene;
	} descriptorSetLayouts;

	// We will be using one single index and vertex buffer
	// containing vertices and indices for all meshes in the scene
	// This allows us to keep memory allocations down
	vks::Buffer vertexBuffer;
	vks::Buffer indexBuffer;

	vk::DescriptorSet descriptorSetScene;

	const aiScene* aScene;

	// Get materials from the assimp scene and map to our scene structures
	void loadMaterials()
	{
		materials.resize(aScene->mNumMaterials);

		for (size_t i = 0; i < materials.size(); i++)
		{
			materials[i] = {};

			aiString name;
			aScene->mMaterials[i]->Get(AI_MATKEY_NAME, name);

			// Properties
			aiColor4D color;
			aScene->mMaterials[i]->Get(AI_MATKEY_COLOR_AMBIENT, color);
			materials[i].properties.ambient = glm::make_vec4(&color.r) + glm::vec4(0.1f);
			aScene->mMaterials[i]->Get(AI_MATKEY_COLOR_DIFFUSE, color);
			materials[i].properties.diffuse = glm::make_vec4(&color.r);
			aScene->mMaterials[i]->Get(AI_MATKEY_COLOR_SPECULAR, color);
			materials[i].properties.specular = glm::make_vec4(&color.r);
			aScene->mMaterials[i]->Get(AI_MATKEY_OPACITY, materials[i].properties.opacity);

			if ((materials[i].properties.opacity) > 0.0f)
				materials[i].properties.specular = glm::vec4(0.0f);

			materials[i].name = name.C_Str();
			std::cout << "Material \"" << materials[i].name << "\"" << std::endl;

			// Textures
			std::string texFormatSuffix;
			vk::Format texFormat;
			// Get supported compressed texture format
			if (vulkanDevice->features.textureCompressionBC) {
				texFormatSuffix = "_bc3_unorm";
				texFormat = vk::Format::eBc3UnormBlock;
			}
			else if (vulkanDevice->features.textureCompressionASTC_LDR) {
				texFormatSuffix = "_astc_8x8_unorm";
				texFormat = vk::Format::eAstc8x8UnormBlock;
			}
			else if (vulkanDevice->features.textureCompressionETC2) {
				texFormatSuffix = "_etc2_unorm";
				texFormat = vk::Format::eEtc2R8G8B8UnormBlock;
			}
			else {
				vks::tools::exitFatal("Device does not support any compressed texture format!", "Error");
			}

			aiString texturefile;
			// Diffuse
			aScene->mMaterials[i]->GetTexture(aiTextureType_DIFFUSE, 0, &texturefile);
			if (aScene->mMaterials[i]->GetTextureCount(aiTextureType_DIFFUSE) > 0)
			{
				std::cout << "  Diffuse: \"" << texturefile.C_Str() << "\"" << std::endl;
				std::string fileName = std::string(texturefile.C_Str());
				std::replace(fileName.begin(), fileName.end(), '\\', '/');
				fileName.insert(fileName.find(".ktx"), texFormatSuffix);
				materials[i].diffuse.loadFromFile(assetPath + fileName, texFormat, vulkanDevice, queue);
			}
			else
			{
				std::cout << "  Material has no diffuse, using dummy texture!" << std::endl;
				// todo : separate pipeline and layout
				materials[i].diffuse.loadFromFile(assetPath + "dummy_rgba_unorm.ktx", vk::Format::eR8G8B8A8Unorm, vulkanDevice, queue);
			}

			// For scenes with multiple textures per material we would need to check for additional texture types, e.g.:
			// aiTextureType_HEIGHT, aiTextureType_OPACITY, aiTextureType_SPECULAR, etc.

			// Assign pipeline
			materials[i].pipeline = (materials[i].properties.opacity == 0.0f) ? &pipelines.solid : &pipelines.blending;
		}

		// Generate descriptor sets for the materials

		// Descriptor pool
		std::vector<vk::DescriptorPoolSize> poolSizes;
		poolSizes.push_back(vks::initializers::descriptorPoolSize(vk::DescriptorType::eUniformBuffer, static_cast<uint32_t>(materials.size())));
		poolSizes.push_back(vks::initializers::descriptorPoolSize(vk::DescriptorType::eCombinedImageSampler, static_cast<uint32_t>(materials.size())));

		vk::DescriptorPoolCreateInfo descriptorPoolInfo =
			vks::initializers::descriptorPoolCreateInfo(
				static_cast<uint32_t>(poolSizes.size()),
				poolSizes.data(),
				static_cast<uint32_t>(materials.size()) + 1);

		descriptorPool = vulkanDevice->logicalDevice.createDescriptorPool(descriptorPoolInfo);

		// Descriptor set and pipeline layouts
		std::vector<vk::DescriptorSetLayoutBinding> setLayoutBindings;
		vk::DescriptorSetLayoutCreateInfo descriptorLayout;

		// Set 0: Scene matrices
		setLayoutBindings.push_back(vks::initializers::descriptorSetLayoutBinding(
			vk::DescriptorType::eUniformBuffer,
			vk::ShaderStageFlagBits::eVertex,
			0));
		descriptorLayout = vks::initializers::descriptorSetLayoutCreateInfo(
				setLayoutBindings);
		descriptorSetLayouts.scene = vulkanDevice->logicalDevice.createDescriptorSetLayout(descriptorLayout);

		// Set 1: Material data
		setLayoutBindings.clear();
		setLayoutBindings.push_back(vks::initializers::descriptorSetLayoutBinding(
			vk::DescriptorType::eCombinedImageSampler,
			vk::ShaderStageFlagBits::eFragment,
			0));
		descriptorSetLayouts.material = vulkanDevice->logicalDevice.createDescriptorSetLayout(descriptorLayout);

		// Setup pipeline layout
		std::array<vk::DescriptorSetLayout, 2> setLayouts = { descriptorSetLayouts.scene, descriptorSetLayouts.material };
		vk::PipelineLayoutCreateInfo pipelineLayoutCreateInfo = vks::initializers::pipelineLayoutCreateInfo(setLayouts.data(), static_cast<uint32_t>(setLayouts.size()));

		// We will be using a push constant block to pass material properties to the fragment shaders
		vk::PushConstantRange pushConstantRange = vks::initializers::pushConstantRange(
			vk::ShaderStageFlagBits::eFragment, 
			sizeof(SceneMaterialProperties), 
			0);
		pipelineLayoutCreateInfo.pushConstantRangeCount = 1;
		pipelineLayoutCreateInfo.pPushConstantRanges = &pushConstantRange;

		pipelineLayout = vulkanDevice->logicalDevice.createPipelineLayout(pipelineLayoutCreateInfo);

		// Material descriptor sets
		for (size_t i = 0; i < materials.size(); i++)
		{
			// Descriptor set
			vk::DescriptorSetAllocateInfo allocInfo =
				vks::initializers::descriptorSetAllocateInfo(
					descriptorPool,
					&descriptorSetLayouts.material,
					1);

			materials[i].descriptorSet = vulkanDevice->logicalDevice.allocateDescriptorSets(allocInfo)[0];

			std::vector<vk::WriteDescriptorSet> writeDescriptorSets;

			// todo : only use image sampler descriptor set and use one scene ubo for matrices

			// Binding 0: Diffuse texture
			writeDescriptorSets.push_back(vks::initializers::writeDescriptorSet(
				materials[i].descriptorSet,
				vk::DescriptorType::eCombinedImageSampler,
				0,
				&materials[i].diffuse.descriptor));

			vulkanDevice->logicalDevice.updateDescriptorSets(writeDescriptorSets, nullptr);
		}

		// Scene descriptor set
		vk::DescriptorSetAllocateInfo allocInfo =
			vks::initializers::descriptorSetAllocateInfo(
				descriptorPool,
				&descriptorSetLayouts.scene,
				1);
		descriptorSetScene = vulkanDevice->logicalDevice.allocateDescriptorSets(allocInfo)[0];

		std::vector<vk::WriteDescriptorSet> writeDescriptorSets;
		// Binding 0 : Vertex shader uniform buffer
		writeDescriptorSets.push_back(vks::initializers::writeDescriptorSet(
			descriptorSetScene,
			vk::DescriptorType::eUniformBuffer,
			0,
			&uniformBuffer.descriptor));

		vulkanDevice->logicalDevice.updateDescriptorSets(writeDescriptorSets, nullptr);
	}

	// Load all meshes from the scene and generate the buffers for rendering them
	void loadMeshes(vk::CommandBuffer copyCmd)
	{
		std::vector<Vertex> vertices;
		std::vector<uint32_t> indices;
		uint32_t indexBase = 0;

		meshes.resize(aScene->mNumMeshes);
		for (uint32_t i = 0; i < meshes.size(); i++)
		{
			aiMesh *aMesh = aScene->mMeshes[i];

			std::cout << "Mesh \"" << aMesh->mName.C_Str() << "\"" << std::endl;
			std::cout << "	Material: \"" << materials[aMesh->mMaterialIndex].name << "\"" << std::endl;
			std::cout << "	Faces: " << aMesh->mNumFaces << std::endl;

			meshes[i].material = &materials[aMesh->mMaterialIndex];
			meshes[i].indexBase = indexBase;
			meshes[i].indexCount = aMesh->mNumFaces * 3;

			// Vertices
			bool hasUV = aMesh->HasTextureCoords(0);
			bool hasColor = aMesh->HasVertexColors(0);
			bool hasNormals = aMesh->HasNormals();

			for (uint32_t v = 0; v < aMesh->mNumVertices; v++)
			{
				Vertex vertex;
				vertex.pos = glm::make_vec3(&aMesh->mVertices[v].x);
				vertex.pos.y = -vertex.pos.y;
				vertex.uv = hasUV ? glm::make_vec2(&aMesh->mTextureCoords[0][v].x) : glm::vec2(0.0f);
				vertex.normal = hasNormals ? glm::make_vec3(&aMesh->mNormals[v].x) : glm::vec3(0.0f);
				vertex.normal.y = -vertex.normal.y;
				vertex.color = hasColor ? glm::make_vec3(&aMesh->mColors[0][v].r) : glm::vec3(1.0f);
				vertices.push_back(vertex);
			}

			// Indices
			for (uint32_t f = 0; f < aMesh->mNumFaces; f++)
			{
				for (uint32_t j = 0; j < 3; j++)
				{
					indices.push_back(aMesh->mFaces[f].mIndices[j]);
				}
			}

			indexBase += aMesh->mNumFaces * 3;
		}

		// Create buffers
		// For better performance we only create one index and vertex buffer to keep number of memory allocations down
		size_t vertexDataSize = vertices.size() * sizeof(Vertex);
		size_t indexDataSize = indices.size() * sizeof(uint32_t);
		
		vks::Buffer vertexStaging, indexStaging;

		// Vertex buffer
		// Staging buffer
		vulkanDevice->createBuffer(
			vk::BufferUsageFlagBits::eTransferSrc,
			vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
			&vertexStaging,
			static_cast<uint32_t>(vertexDataSize),
			vertices.data());
		// Target
		vulkanDevice->createBuffer(
			vk::BufferUsageFlagBits::eVertexBuffer | vk::BufferUsageFlagBits::eTransferDst,
			vk::MemoryPropertyFlagBits::eDeviceLocal,
			&vertexBuffer,
			static_cast<uint32_t>(vertexDataSize));

		// Index buffer
		vulkanDevice->createBuffer(
			vk::BufferUsageFlagBits::eTransferSrc,
			vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
			&indexStaging,
			static_cast<uint32_t>(indexDataSize),
			indices.data());
		// Target
		vulkanDevice->createBuffer(
			vk::BufferUsageFlagBits::eIndexBuffer | vk::BufferUsageFlagBits::eTransferDst,
			vk::MemoryPropertyFlagBits::eDeviceLocal,
			&indexBuffer,
			static_cast<uint32_t>(indexDataSize));

		// Copy
		vk::CommandBufferBeginInfo cmdBufInfo = vks::initializers::commandBufferBeginInfo();
		copyCmd.begin(cmdBufInfo);

		vk::BufferCopy copyRegion = {};

		copyRegion.size = vertexDataSize;
		copyCmd.copyBuffer(
			vertexStaging.buffer,
			vertexBuffer.buffer,
			copyRegion);

		copyRegion.size = indexDataSize;
		copyCmd.copyBuffer(
			indexStaging.buffer,
			indexBuffer.buffer,
			copyRegion);

		copyCmd.end();

		vk::SubmitInfo submitInfo = {};

		submitInfo.commandBufferCount = 1;
		submitInfo.pCommandBuffers = &copyCmd;

		queue.submit(submitInfo, vk::Fence(nullptr));
		queue.waitIdle();

		//todo: fence
		vertexStaging.destroy();
		indexStaging.destroy();
	}

public:
#if defined(__ANDROID__)
	AAssetManager* assetManager = nullptr;
#endif

	std::string assetPath = "";

	std::vector<SceneMaterial> materials;
	std::vector<ScenePart> meshes;

	// Shared ubo containing matrices used by all
	// materials and meshes
	vks::Buffer uniformBuffer;
	struct UniformData {
		glm::mat4 projection;
		glm::mat4 view;
		glm::mat4 model;
		glm::vec4 lightPos = glm::vec4(1.25f, 8.35f, 0.0f, 0.0f);
	} uniformData;

	// Scene uses multiple pipelines
	struct {
		vk::Pipeline solid;
		vk::Pipeline blending;
		vk::Pipeline wireframe;
	} pipelines;

	// Shared pipeline layout
	vk::PipelineLayout pipelineLayout;

	// For displaying only a single part of the scene
	bool renderSingleScenePart = false;
	uint32_t scenePartIndex = 0;

	// Default constructor
	Scene(vks::VulkanDevice *vulkanDevice, vk::Queue queue)
	{
		this->vulkanDevice = vulkanDevice;
		this->queue = queue;

		// Prepare uniform buffer for global matrices
		vk::MemoryRequirements memReqs;
		vk::MemoryAllocateInfo memAlloc = vks::initializers::memoryAllocateInfo();
		vk::BufferCreateInfo bufferCreateInfo = vks::initializers::bufferCreateInfo(vk::BufferUsageFlagBits::eUniformBuffer, sizeof(uniformData));
		uniformBuffer.buffer = vulkanDevice->logicalDevice.createBuffer(bufferCreateInfo);
		memReqs = vulkanDevice->logicalDevice.getBufferMemoryRequirements(uniformBuffer.buffer);
		memAlloc.allocationSize = memReqs.size;
		memAlloc.memoryTypeIndex = vulkanDevice->getMemoryType(memReqs.memoryTypeBits, vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);
		uniformBuffer.memory = vulkanDevice->logicalDevice.allocateMemory(memAlloc);
		vulkanDevice->logicalDevice.bindBufferMemory(uniformBuffer.buffer, uniformBuffer.memory, 0);
		uniformBuffer.mapped = vulkanDevice->logicalDevice.mapMemory(uniformBuffer.memory, 0, sizeof(uniformData), vk::MemoryMapFlags());
		uniformBuffer.descriptor.offset = 0;
		uniformBuffer.descriptor.buffer = uniformBuffer.buffer;
		uniformBuffer.descriptor.range = sizeof(uniformData);
		uniformBuffer.device = vulkanDevice->logicalDevice;
	}

	// Default destructor
	~Scene()
	{
		vertexBuffer.destroy();
		indexBuffer.destroy();
		for (auto material : materials)
		{
			material.diffuse.destroy();
		}
		vulkanDevice->logicalDevice.destroyPipelineLayout(pipelineLayout);
		vulkanDevice->logicalDevice.destroyDescriptorSetLayout(descriptorSetLayouts.material);
		vulkanDevice->logicalDevice.destroyDescriptorSetLayout(descriptorSetLayouts.scene);
		vulkanDevice->logicalDevice.destroyDescriptorPool(descriptorPool);
		vulkanDevice->logicalDevice.destroyPipeline(pipelines.solid);
		vulkanDevice->logicalDevice.destroyPipeline(pipelines.blending);
		vulkanDevice->logicalDevice.destroyPipeline(pipelines.wireframe);
		uniformBuffer.destroy();
	}

	void load(std::string filename, vk::CommandBuffer copyCmd)
	{
		Assimp::Importer Importer;

		int flags = aiProcess_PreTransformVertices | aiProcess_Triangulate | aiProcess_GenNormals;

#if defined(__ANDROID__)
		AAsset* asset = AAssetManager_open(assetManager, filename.c_str(), AASSET_MODE_STREAMING);
		assert(asset);
		size_t size = AAsset_getLength(asset);
		assert(size > 0);
		void *meshData = malloc(size);
		AAsset_read(asset, meshData, size);
		AAsset_close(asset);
		aScene = Importer.ReadFileFromMemory(meshData, size, flags);
		free(meshData);
#else
		aScene = Importer.ReadFile(filename.c_str(), flags);
#endif
		if (aScene)
		{
			loadMaterials();
			loadMeshes(copyCmd);
		}
		else
		{
			printf("Error parsing '%s': '%s'\n", filename.c_str(), Importer.GetErrorString());
#if defined(__ANDROID__)
			LOGE("Error parsing '%s': '%s'", filename.c_str(), Importer.GetErrorString());
#endif
		}

	}

	// Renders the scene into an active command buffer
	// In a real world application we would do some visibility culling in here
	void render(vk::CommandBuffer cmdBuffer, bool wireframe)
	{
		std::vector<vk::DeviceSize> offsets = { 0 };

		// Bind scene vertex and index buffers
		cmdBuffer.bindVertexBuffers(0, vertexBuffer.buffer, offsets);
		cmdBuffer.bindIndexBuffer(indexBuffer.buffer, 0, vk::IndexType::eUint32);

		for (size_t i = 0; i < meshes.size(); i++)
		{
			if ((renderSingleScenePart) && (i != scenePartIndex))
				continue;

			// todo : per material pipelines
			// cmdBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, *mesh.material->pipeline);

			// We will be using multiple descriptor sets for rendering
			// In GLSL the selection is done via the set and binding keywords
			// VS: layout (set = 0, binding = 0) uniform UBO;
			// FS: layout (set = 1, binding = 0) uniform sampler2D samplerColorMap;

			std::array<vk::DescriptorSet, 2> descriptorSets;
			// Set 0: Scene descriptor set containing global matrices
			descriptorSets[0] = descriptorSetScene;
			// Set 1: Per-Material descriptor set containing bound images
			descriptorSets[1] = meshes[i].material->descriptorSet;

			cmdBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, wireframe ? pipelines.wireframe : *meshes[i].material->pipeline);
			cmdBuffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, pipelineLayout, 0, descriptorSets, nullptr);

			// Pass material properies via push constants
			cmdBuffer.pushConstants(
				pipelineLayout,
				vk::ShaderStageFlagBits::eFragment,
				0,
				sizeof(SceneMaterialProperties),
				&meshes[i].material->properties);

			// Render from the global scene vertex buffer using the mesh index offset
			cmdBuffer.drawIndexed(meshes[i].indexCount, 1, 0, meshes[i].indexBase, 0);
		}
	}
};

class VulkanExample : public VulkanExampleBase
{
public:
	bool wireframe = false;
	bool attachLight = false;

	Scene *scene = nullptr;

	struct {
		vk::PipelineVertexInputStateCreateInfo inputState;
		std::vector<vk::VertexInputBindingDescription> bindingDescriptions;
		std::vector<vk::VertexInputAttributeDescription> attributeDescriptions;
	} vertices;

	VulkanExample() : VulkanExampleBase(ENABLE_VALIDATION)
	{
		title = "Vulkan Example - Scene rendering";
		rotationSpeed = 0.5f;
		enableTextOverlay = true;
		camera.type = Camera::CameraType::firstperson;
		camera.movementSpeed = 7.5f;
		camera.position = { 15.0f, -13.5f, 0.0f };
		camera.setRotation(glm::vec3(5.0f, 90.0f, 0.0f));
		camera.setPerspective(60.0f, (float)width / (float)height, 0.1f, 256.0f);
	}

	~VulkanExample()
	{
		delete(scene);
	}

	// Enable physical device features required for this example				
	virtual void getEnabledFeatures()
	{
		// Fill mode non solid is required for wireframe display
		if (deviceFeatures.fillModeNonSolid) {
			enabledFeatures.fillModeNonSolid = VK_TRUE;
		};
	}

	void reBuildCommandBuffers()
	{
		if (!checkCommandBuffers())
		{
			destroyCommandBuffers();
			createCommandBuffers();
		}
		buildCommandBuffers();
	}

	void buildCommandBuffers()
	{
		vk::CommandBufferBeginInfo cmdBufInfo = vks::initializers::commandBufferBeginInfo();

		vk::ClearValue clearValues[2];
		clearValues[0].color = defaultClearColor;
		clearValues[0].color = vk::ClearColorValue{ std::array<float, 4>{ 0.25f, 0.25f, 0.25f, 1.0f} };
		clearValues[1].depthStencil = vk::ClearDepthStencilValue{ 1.0f, 0 };

		vk::RenderPassBeginInfo renderPassBeginInfo = vks::initializers::renderPassBeginInfo();
		renderPassBeginInfo.renderPass = renderPass;
		renderPassBeginInfo.renderArea.offset.x = 0;
		renderPassBeginInfo.renderArea.offset.y = 0;
		renderPassBeginInfo.renderArea.extent.width = width;
		renderPassBeginInfo.renderArea.extent.height = height;
		renderPassBeginInfo.clearValueCount = 2;
		renderPassBeginInfo.pClearValues = clearValues;

		for (uint32_t i = 0; i < drawCmdBuffers.size(); ++i)
		{
			renderPassBeginInfo.framebuffer = frameBuffers[i];

			drawCmdBuffers[i].begin(cmdBufInfo);

			drawCmdBuffers[i].beginRenderPass(renderPassBeginInfo, vk::SubpassContents::eInline);

			vk::Viewport viewport = vks::initializers::viewport((float)width, (float)height, 0.0f, 1.0f);
			drawCmdBuffers[i].setViewport(0, viewport);

			vk::Rect2D scissor = vks::initializers::rect2D(width, height, 0, 0);
			drawCmdBuffers[i].setScissor(0, scissor);

			scene->render(drawCmdBuffers[i], wireframe);

			drawCmdBuffers[i].endRenderPass();

			drawCmdBuffers[i].end();
		}
	}

	void setupVertexDescriptions()
	{
		// Binding description
		vertices.bindingDescriptions.resize(1);
		vertices.bindingDescriptions[0] =
			vks::initializers::vertexInputBindingDescription(
				VERTEX_BUFFER_BIND_ID,
				sizeof(Vertex),
				vk::VertexInputRate::eVertex);

		// Attribute descriptions
		// Describes memory layout and shader positions
		vertices.attributeDescriptions.resize(4);
		// Location 0 : Position
		vertices.attributeDescriptions[0] =
			vks::initializers::vertexInputAttributeDescription(
				VERTEX_BUFFER_BIND_ID,
				0,
				vk::Format::eR32G32B32Sfloat,
				0);
		// Location 1 : Normal
		vertices.attributeDescriptions[1] =
			vks::initializers::vertexInputAttributeDescription(
				VERTEX_BUFFER_BIND_ID,
				1,
				vk::Format::eR32G32B32Sfloat,
				sizeof(float) * 3);
		// Location 2 : Texture coordinates
		vertices.attributeDescriptions[2] =
			vks::initializers::vertexInputAttributeDescription(
				VERTEX_BUFFER_BIND_ID,
				2,
				vk::Format::eR32G32Sfloat,
				sizeof(float) * 6);
		// Location 3 : Color
		vertices.attributeDescriptions[3] =
			vks::initializers::vertexInputAttributeDescription(
				VERTEX_BUFFER_BIND_ID,
				3,
				vk::Format::eR32G32B32Sfloat,
				sizeof(float) * 8);

		vertices.inputState = vks::initializers::pipelineVertexInputStateCreateInfo();
		vertices.inputState.vertexBindingDescriptionCount = static_cast<uint32_t>(vertices.bindingDescriptions.size());
		vertices.inputState.pVertexBindingDescriptions = vertices.bindingDescriptions.data();
		vertices.inputState.vertexAttributeDescriptionCount = static_cast<uint32_t>(vertices.attributeDescriptions.size());
		vertices.inputState.pVertexAttributeDescriptions = vertices.attributeDescriptions.data();
	}

	void preparePipelines()
	{
		vk::PipelineInputAssemblyStateCreateInfo inputAssemblyState =
			vks::initializers::pipelineInputAssemblyStateCreateInfo(
				vk::PrimitiveTopology::eTriangleList,
				vk::PipelineInputAssemblyStateCreateFlags(),
				VK_FALSE);

		vk::PipelineRasterizationStateCreateInfo rasterizationState =
			vks::initializers::pipelineRasterizationStateCreateInfo(
				vk::PolygonMode::eFill,
				vk::CullModeFlagBits::eBack,
				vk::FrontFace::eCounterClockwise);

		vk::PipelineColorBlendAttachmentState blendAttachmentState =
			vks::initializers::pipelineColorBlendAttachmentState(
				vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG | vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA,
				VK_FALSE);

		vk::PipelineColorBlendStateCreateInfo colorBlendState =
			vks::initializers::pipelineColorBlendStateCreateInfo(
				1,
				&blendAttachmentState);

		vk::PipelineDepthStencilStateCreateInfo depthStencilState =
			vks::initializers::pipelineDepthStencilStateCreateInfo(
				VK_TRUE,
				VK_TRUE,
				vk::CompareOp::eLessOrEqual);

		vk::PipelineViewportStateCreateInfo viewportState =
			vks::initializers::pipelineViewportStateCreateInfo(1, 1);

		vk::PipelineMultisampleStateCreateInfo multisampleState =
			vks::initializers::pipelineMultisampleStateCreateInfo(vk::SampleCountFlagBits::e1);

		std::vector<vk::DynamicState> dynamicStateEnables = {
			vk::DynamicState::eViewport,
			vk::DynamicState::eScissor
		};
		vk::PipelineDynamicStateCreateInfo dynamicState =
			vks::initializers::pipelineDynamicStateCreateInfo(
				dynamicStateEnables);

		std::array<vk::PipelineShaderStageCreateInfo, 2> shaderStages;

		// Solid rendering pipeline
		shaderStages[0] = loadShader(getAssetPath() + "shaders/scenerendering/scene.vert.spv", vk::ShaderStageFlagBits::eVertex);
		shaderStages[1] = loadShader(getAssetPath() + "shaders/scenerendering/scene.frag.spv", vk::ShaderStageFlagBits::eFragment);

		vk::GraphicsPipelineCreateInfo pipelineCreateInfo =
			vks::initializers::pipelineCreateInfo(
				scene->pipelineLayout,
				renderPass);

		pipelineCreateInfo.pVertexInputState = &vertices.inputState;
		pipelineCreateInfo.pInputAssemblyState = &inputAssemblyState;
		pipelineCreateInfo.pRasterizationState = &rasterizationState;
		pipelineCreateInfo.pColorBlendState = &colorBlendState;
		pipelineCreateInfo.pMultisampleState = &multisampleState;
		pipelineCreateInfo.pViewportState = &viewportState;
		pipelineCreateInfo.pDepthStencilState = &depthStencilState;
		pipelineCreateInfo.pDynamicState = &dynamicState;
		pipelineCreateInfo.stageCount = static_cast<uint32_t>(shaderStages.size());
		pipelineCreateInfo.pStages = shaderStages.data();

		scene->pipelines.solid = device.createGraphicsPipelines(pipelineCache, pipelineCreateInfo)[0];

		// Alpha blended pipeline
		rasterizationState.cullMode = vk::CullModeFlagBits::eNone;
		blendAttachmentState.blendEnable = VK_TRUE;
		blendAttachmentState.colorBlendOp = vk::BlendOp::eAdd;
		blendAttachmentState.srcColorBlendFactor = vk::BlendFactor::eSrcColor;
		blendAttachmentState.dstColorBlendFactor = vk::BlendFactor::eOneMinusSrcColor;

		scene->pipelines.blending = device.createGraphicsPipelines(pipelineCache, pipelineCreateInfo)[0];

		// Wire frame rendering pipeline
		if (deviceFeatures.fillModeNonSolid) {
			rasterizationState.cullMode = vk::CullModeFlagBits::eBack;
			blendAttachmentState.blendEnable = VK_FALSE;
			rasterizationState.polygonMode = vk::PolygonMode::eLine;
			rasterizationState.lineWidth = 1.0f;
			scene->pipelines.wireframe = device.createGraphicsPipelines(pipelineCache, pipelineCreateInfo)[0];
		}
	}

	void updateUniformBuffers()
	{
		if (attachLight)
		{
			scene->uniformData.lightPos = glm::vec4(-camera.position, 1.0f);
		}

		scene->uniformData.projection = camera.matrices.perspective;
		scene->uniformData.view = camera.matrices.view;
		scene->uniformData.model = glm::mat4();

		memcpy(scene->uniformBuffer.mapped, &scene->uniformData, sizeof(scene->uniformData));
	}

	void draw()
	{
		VulkanExampleBase::prepareFrame();

		// Command buffer to be sumitted to the queue
		submitInfo.commandBufferCount = 1;
		submitInfo.pCommandBuffers = &drawCmdBuffers[currentBuffer];

		// Submit to queue
		queue.submit(submitInfo, vk::Fence(nullptr));

		VulkanExampleBase::submitFrame();
	}

	void loadScene()
	{
		vk::CommandBuffer copyCmd = VulkanExampleBase::createCommandBuffer(vk::CommandBufferLevel::ePrimary, false);
		scene = new Scene(vulkanDevice, queue);

#if defined(__ANDROID__)
		scene->assetManager = androidApp->activity->assetManager;
#endif
		scene->assetPath = getAssetPath() + "models/sibenik/";
		scene->load(getAssetPath() + "models/sibenik/sibenik.dae", copyCmd);
		device.freeCommandBuffers(cmdPool, copyCmd);
		updateUniformBuffers();
	}

	void prepare()
	{
		VulkanExampleBase::prepare();
		setupVertexDescriptions();
		loadScene();
		preparePipelines();
		buildCommandBuffers();
		prepared = true;
	}

	virtual void render()
	{
		if (!prepared)
			return;
		draw();
	}

	virtual void viewChanged()
	{
		updateUniformBuffers();
	}

	virtual void keyPressed(uint32_t keyCode)
	{
		switch (keyCode)
		{
		case KEY_SPACE:
		case GAMEPAD_BUTTON_A:
			if (deviceFeatures.fillModeNonSolid) {
				wireframe = !wireframe;
				reBuildCommandBuffers();
			}
			break;
		case KEY_P:
			scene->renderSingleScenePart = !scene->renderSingleScenePart;
			reBuildCommandBuffers();
			updateTextOverlay();
			break;
		case KEY_KPADD:
			scene->scenePartIndex = (scene->scenePartIndex < static_cast<uint32_t>(scene->meshes.size())) ? scene->scenePartIndex + 1 : 0;
			reBuildCommandBuffers();
			updateTextOverlay();
			break;
		case KEY_KPSUB:
			scene->scenePartIndex = (scene->scenePartIndex > 0) ? scene->scenePartIndex - 1 : static_cast<uint32_t>(scene->meshes.size()) - 1;
			updateTextOverlay();
			reBuildCommandBuffers();
			break;
		case KEY_L:
			attachLight = !attachLight;
			updateUniformBuffers();
			break;
		}
	}

	virtual void getOverlayText(VulkanTextOverlay *textOverlay)
	{
		if (deviceFeatures.fillModeNonSolid) {
#if defined(__ANDROID__)
			textOverlay->addText("Press \"Button A\" to toggle wireframe", 5.0f, 85.0f, VulkanTextOverlay::alignLeft);
#else
			textOverlay->addText("Press \"space\" to toggle wireframe", 5.0f, 85.0f, VulkanTextOverlay::alignLeft);
			if ((scene) && (scene->renderSingleScenePart))
			{
				textOverlay->addText("Rendering mesh " + std::to_string(scene->scenePartIndex + 1) + " of " + std::to_string(static_cast<uint32_t>(scene->meshes.size())) + "(\"p\" to toggle)", 5.0f, 100.0f, VulkanTextOverlay::alignLeft);
			}
			else
			{
				textOverlay->addText("Rendering whole scene (\"p\" to toggle)", 5.0f, 100.0f, VulkanTextOverlay::alignLeft);
			}
#endif
		}
	}
};

VULKAN_EXAMPLE_MAIN()