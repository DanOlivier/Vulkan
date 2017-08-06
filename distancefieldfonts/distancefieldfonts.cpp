/*
* Vulkan Example - Font rendering using signed distance fields
*
* Font generated using https://github.com/libgdx/libgdx/wiki/Hiero
*
* Copyright (C) 2016 by Sascha Willems - www.saschawillems.de
*
* This code is licensed under the MIT license (MIT) (http://opensource.org/licenses/MIT)
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sstream>
#include <assert.h>
#include <vector>
#include <array>

#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <vulkan/vulkan.hpp>
#include "vulkanexamplebase.h"
#include "VulkanTexture.hpp"
#include "VulkanBuffer.hpp"

#define VERTEX_BUFFER_BIND_ID 0
#define ENABLE_VALIDATION false

// Vertex layout for this example
struct Vertex {
	float pos[3];
	float uv[2];
};

// AngelCode .fnt format structs and classes
struct bmchar {
	uint32_t x, y;
	uint32_t width;
	uint32_t height;
	int32_t xoffset;
	int32_t yoffset;
	int32_t xadvance;
	uint32_t page;
};

// Quick and dirty : complete ASCII table
// Only chars present in the .fnt are filled with data!
std::array<bmchar, 255> fontChars;

int32_t nextValuePair(std::stringstream *stream)
{
	std::string pair;
	*stream >> pair;
	uint32_t spos = pair.find("=");
	std::string value = pair.substr(spos + 1);
	int32_t val = std::stoi(value);
	return val;
}

class VulkanExample : public VulkanExampleBase
{
public:
	bool splitScreen = true;

	struct {
		vks::Texture2D fontSDF;
		vks::Texture2D fontBitmap;
	} textures;

	struct {
		vk::PipelineVertexInputStateCreateInfo inputState;
		std::vector<vk::VertexInputBindingDescription> bindingDescriptions;
		std::vector<vk::VertexInputAttributeDescription> attributeDescriptions;
	} vertices;

	vks::Buffer vertexBuffer;
	vks::Buffer indexBuffer;
	uint32_t indexCount;

	struct {
		vks::Buffer vs;
		vks::Buffer fs;
	} uniformBuffers;

	struct UBOVS {
		glm::mat4 projection;
		glm::mat4 model;
	} uboVS;

	struct UBOFS {
		glm::vec4 outlineColor = glm::vec4(1.0f, 0.0f, 0.0f, 0.0f);
		float outlineWidth = 0.6f;
		float outline = true;
	} uboFS;

	struct {
		vk::Pipeline sdf;
		vk::Pipeline bitmap;
	} pipelines;

	struct {
		vk::DescriptorSet sdf;
		vk::DescriptorSet bitmap;
	} descriptorSets;

	vk::PipelineLayout pipelineLayout;
	vk::DescriptorSetLayout descriptorSetLayout;

	VulkanExample() : VulkanExampleBase(ENABLE_VALIDATION)
	{
		zoom = -2.0f;
		enableTextOverlay = true;
		title = "Vulkan Example - Distance field fonts";
	}

	~VulkanExample()
	{
		// Clean up used Vulkan resources 
		// Note : Inherited destructor cleans up resources stored in base class

		// Clean up texture resources
		textures.fontSDF.destroy();
		textures.fontBitmap.destroy();

		device.destroyPipeline(pipelines.sdf);
		device.destroyPipeline(pipelines.bitmap);

		device.destroyPipelineLayout(pipelineLayout);
		device.destroyDescriptorSetLayout(descriptorSetLayout);

		vertexBuffer.destroy();
		indexBuffer.destroy();

		uniformBuffers.vs.destroy();
		uniformBuffers.fs.destroy();
	}

	// Basic parser fpr AngelCode bitmap font format files
	// See http://www.angelcode.com/products/bmfont/doc/file_format.html for details
	void parsebmFont()
	{
		std::string fileName = getAssetPath() + "font.fnt";

#if defined(__ANDROID__)
		// Font description file is stored inside the apk
		// So we need to load it using the asset manager
		AAsset* asset = AAssetManager_open(androidApp->activity->assetManager, fileName.c_str(), AASSET_MODE_STREAMING);
		assert(asset);
		size_t size = AAsset_getLength(asset);

		assert(size > 0);

		void *fileData = malloc(size);
		AAsset_read(asset, fileData, size);
		AAsset_close(asset);

		std::stringbuf sbuf((const char*)fileData);
		std::istream istream(&sbuf);
#else
		std::filebuf fileBuffer;
		fileBuffer.open(fileName, std::ios::in);
		std::istream istream(&fileBuffer);
#endif

		assert(istream.good());

		while (!istream.eof())
		{
			std::string line;
			std::stringstream lineStream;
			std::getline(istream, line);
			lineStream << line;

			std::string info;
			lineStream >> info;

			if (info == "char")
			{
				std::string pair;

				// char id
				uint32_t charid = nextValuePair(&lineStream);
				// Char properties
				fontChars[charid].x = nextValuePair(&lineStream);
				fontChars[charid].y = nextValuePair(&lineStream);
				fontChars[charid].width = nextValuePair(&lineStream);
				fontChars[charid].height = nextValuePair(&lineStream);
				fontChars[charid].xoffset = nextValuePair(&lineStream);
				fontChars[charid].yoffset = nextValuePair(&lineStream);
				fontChars[charid].xadvance = nextValuePair(&lineStream);
				fontChars[charid].page = nextValuePair(&lineStream);
			}
		}

	}

	void loadAssets()
	{
		textures.fontSDF.loadFromFile(getAssetPath() + "textures/font_sdf_rgba.ktx", vk::Format::eR8G8B8A8Unorm, vulkanDevice, queue);
		textures.fontBitmap.loadFromFile(getAssetPath() + "textures/font_bitmap_rgba.ktx", vk::Format::eR8G8B8A8Unorm, vulkanDevice, queue);
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
		clearValues[1].depthStencil = { 1.0f, 0 };

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

			vk::Viewport viewport = vks::initializers::viewport((float)width, (splitScreen) ? (float)height / 2.0f : (float)height, 0.0f, 1.0f);
			drawCmdBuffers[i].setViewport(0, viewport);

			vk::Rect2D scissor = vks::initializers::rect2D(width, height, 0, 0);
			vkCmdSetScissor(drawCmdBuffers[i], 0, 1, &scissor);

			vk::DeviceSize offsets[1] = { 0 };

			// Signed distance field font
			drawCmdBuffers[i].bindDescriptorSets(vk::PipelineBindPoint::eGraphics, pipelineLayout, 0, descriptorSets.sdf, nullptr);
			drawCmdBuffers[i].bindPipeline(vk::PipelineBindPoint::eGraphics, pipelines.sdf);
			drawCmdBuffers[i].bindVertexBuffers(VERTEX_BUFFER_BIND_ID, vertexBuffer.buffer, offsets);
			drawCmdBuffers[i].bindIndexBuffer(indexBuffer.buffer, 0, vk::IndexType::eUint32);
			drawCmdBuffers[i].drawIndexed(indexCount, 1, 0, 0, 0);

			// Linear filtered bitmap font
			if (splitScreen)
			{
				viewport.y = (float)height / 2.0f;
				drawCmdBuffers[i].setViewport(0, viewport);
				drawCmdBuffers[i].bindDescriptorSets(vk::PipelineBindPoint::eGraphics, pipelineLayout, 0, descriptorSets.bitmap, nullptr);
				drawCmdBuffers[i].bindPipeline(vk::PipelineBindPoint::eGraphics, pipelines.bitmap);
				drawCmdBuffers[i].drawIndexed(indexCount, 1, 0, 0, 0);
			}

			drawCmdBuffers[i].endRenderPass();

			drawCmdBuffers[i].end();
		}
	}

	// Creates a vertex buffer containing quads for the passed text
	void generateText(std:: string text)
	{
		std::vector<Vertex> vertices;
		std::vector<uint32_t> indices;
		uint32_t indexOffset = 0;

		float w = textures.fontSDF.width;

		float posx = 0.0f;
		float posy = 0.0f;

		for (uint32_t i = 0; i < text.size(); i++)
		{
			bmchar *charInfo = &fontChars[(int)text[i]];

			if (charInfo->width == 0)
				charInfo->width = 36;

			float charw = ((float)(charInfo->width) / 36.0f);
			float dimx = 1.0f * charw;
			float charh = ((float)(charInfo->height) / 36.0f);
			float dimy = 1.0f * charh;
			posy = 1.0f - charh;

			float us = charInfo->x / w;
			float ue = (charInfo->x + charInfo->width) / w;
			float ts = charInfo->y / w;
			float te = (charInfo->y + charInfo->height) / w;

			float xo = charInfo->xoffset / 36.0f;
			//float yo = charInfo->yoffset / 36.0f;

			vertices.push_back({ { posx + dimx + xo,  posy + dimy, 0.0f }, { ue, te } });
			vertices.push_back({ { posx + xo,         posy + dimy, 0.0f }, { us, te } });
			vertices.push_back({ { posx + xo,         posy,        0.0f }, { us, ts } });
			vertices.push_back({ { posx + dimx + xo,  posy,        0.0f }, { ue, ts } });

			std::array<uint32_t, 6> letterIndices = { 0,1,2, 2,3,0 };
			for (auto& index : letterIndices)
			{
				indices.push_back(indexOffset + index);
			}
			indexOffset += 4;

			float advance = ((float)(charInfo->xadvance) / 36.0f);
			posx += advance;
		}
		indexCount = indices.size();

		// Center
		for (auto& v : vertices)
		{
			v.pos[0] -= posx / 2.0f;
			v.pos[1] -= 0.5f;
		}

		// Generate host accesible buffers for the text vertices and indices and upload the data

		VK_CHECK_RESULT(vulkanDevice->createBuffer(
			vk::BufferUsageFlagBits::eVertexBuffer,
			vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
			&vertexBuffer,
			vertices.size() * sizeof(Vertex),
			vertices.data()));

		VK_CHECK_RESULT(vulkanDevice->createBuffer(
			vk::BufferUsageFlagBits::eIndexBuffer,
			vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
			&indexBuffer,
			indices.size() * sizeof(uint32_t),
			indices.data()));
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
		vertices.attributeDescriptions.resize(2);
		// Location 0 : Position
		vertices.attributeDescriptions[0] =
			vks::initializers::vertexInputAttributeDescription(
				VERTEX_BUFFER_BIND_ID,
				0,
				vk::Format::eR32G32B32Sfloat,
				0);			
		// Location 1 : Texture coordinates
		vertices.attributeDescriptions[1] =
			vks::initializers::vertexInputAttributeDescription(
				VERTEX_BUFFER_BIND_ID,
				1,
				vk::Format::eR32G32Sfloat,
				sizeof(float) * 3);

		vertices.inputState = vks::initializers::pipelineVertexInputStateCreateInfo();
		vertices.inputState.vertexBindingDescriptionCount = vertices.bindingDescriptions.size();
		vertices.inputState.pVertexBindingDescriptions = vertices.bindingDescriptions.data();
		vertices.inputState.vertexAttributeDescriptionCount = vertices.attributeDescriptions.size();
		vertices.inputState.pVertexAttributeDescriptions = vertices.attributeDescriptions.data();
	}

	void setupDescriptorPool()
	{
		std::vector<vk::DescriptorPoolSize> poolSizes =
		{
			vks::initializers::descriptorPoolSize(vk::DescriptorType::eUniformBuffer, 4),
			vks::initializers::descriptorPoolSize(vk::DescriptorType::eCombinedImageSampler, 2)
		};

		vk::DescriptorPoolCreateInfo descriptorPoolInfo = 
			vks::initializers::descriptorPoolCreateInfo(
				poolSizes.size(),
				poolSizes.data(),
				2);

		vk::Result vkRes = vkCreateDescriptorPool(device, &descriptorPoolInfo, nullptr, &descriptorPool);
		assert(!vkRes);
	}

	void setupDescriptorSetLayout()
	{
		std::vector<vk::DescriptorSetLayoutBinding> setLayoutBindings = 
		{
			// Binding 0 : Vertex shader uniform buffer
			vks::initializers::descriptorSetLayoutBinding(
				vk::DescriptorType::eUniformBuffer, 
				vk::ShaderStageFlagBits::eVertex, 
				0),
			// Binding 1 : Fragment shader image sampler
			vks::initializers::descriptorSetLayoutBinding(
				vk::DescriptorType::eCombinedImageSampler, 
				vk::ShaderStageFlagBits::eFragment, 
				1),
			// Binding 2 : Fragment shader uniform buffer
			vks::initializers::descriptorSetLayoutBinding(
				vk::DescriptorType::eUniformBuffer,
				vk::ShaderStageFlagBits::eFragment,
				2)
		};

		vk::DescriptorSetLayoutCreateInfo descriptorLayout = 
			vks::initializers::descriptorSetLayoutCreateInfo(
				setLayoutBindings.data(),
				setLayoutBindings.size());

		descriptorSetLayout = device.createDescriptorSetLayout(descriptorLayout);

		vk::PipelineLayoutCreateInfo pPipelineLayoutCreateInfo =
			vks::initializers::pipelineLayoutCreateInfo(
				&descriptorSetLayout,
				1);

		pipelineLayout = device.createPipelineLayout(pPipelineLayoutCreateInfo);
	}

	void setupDescriptorSet()
	{
		vk::DescriptorSetAllocateInfo allocInfo = 
			vks::initializers::descriptorSetAllocateInfo(
				descriptorPool,
				&descriptorSetLayout,
				1);

		// Signed distance front descriptor set
		descriptorSets.sdf = device.allocateDescriptorSets(allocInfo)[0];

		// Image descriptor for the color map texture
		vk::DescriptorImageInfo texDescriptor =
			vks::initializers::descriptorImageInfo(
				textures.fontSDF.sampler,
				textures.fontSDF.view,
				vk::ImageLayout::eGeneral);

		std::vector<vk::WriteDescriptorSet> writeDescriptorSets =
		{
			// Binding 0 : Vertex shader uniform buffer
			vks::initializers::writeDescriptorSet(
			descriptorSets.sdf,
				vk::DescriptorType::eUniformBuffer, 
				0, 
				&uniformBuffers.vs.descriptor),
			// Binding 1 : Fragment shader texture sampler
			vks::initializers::writeDescriptorSet(
				descriptorSets.sdf,
				vk::DescriptorType::eCombinedImageSampler, 
				1, 
				&texDescriptor),
			// Binding 2 : Fragment shader uniform buffer
			vks::initializers::writeDescriptorSet(
				descriptorSets.sdf,
				vk::DescriptorType::eUniformBuffer,
				2,
				&uniformBuffers.fs.descriptor)
		};

		vkUpdateDescriptorSets(device, writeDescriptorSets.size(), writeDescriptorSets.data(), 0, NULL);

		// Default font rendering descriptor set
		descriptorSets.bitmap = device.allocateDescriptorSets(allocInfo)[0];

		// Image descriptor for the color map texture
		texDescriptor.sampler = textures.fontBitmap.sampler;
		texDescriptor.imageView = textures.fontBitmap.view;

		writeDescriptorSets =
		{
			// Binding 0 : Vertex shader uniform buffer
			vks::initializers::writeDescriptorSet(
				descriptorSets.bitmap,
				vk::DescriptorType::eUniformBuffer,
				0,
				&uniformBuffers.vs.descriptor),
			// Binding 1 : Fragment shader texture sampler
			vks::initializers::writeDescriptorSet(
				descriptorSets.bitmap,
				vk::DescriptorType::eCombinedImageSampler,
				1,
				&texDescriptor)
		};

		vkUpdateDescriptorSets(device, writeDescriptorSets.size(), writeDescriptorSets.data(), 0, NULL);
	}

	void preparePipelines()
	{
		vk::PipelineInputAssemblyStateCreateInfo inputAssemblyState =
			vks::initializers::pipelineInputAssemblyStateCreateInfo(
				vk::PrimitiveTopology::eTriangleList,
				0,
				VK_FALSE);

		vk::PipelineRasterizationStateCreateInfo rasterizationState =
			vks::initializers::pipelineRasterizationStateCreateInfo(
				vk::PolygonMode::eFill,
				vk::CullModeFlagBits::eNone,
				vk::FrontFace::eCounterClockwise,
				0);

		vk::PipelineColorBlendAttachmentState blendAttachmentState =
			vks::initializers::pipelineColorBlendAttachmentState(
				0xf,
				VK_TRUE);

		blendAttachmentState.blendEnable = VK_TRUE;
		blendAttachmentState.srcColorBlendFactor = vk::BlendFactor::eOne;
		blendAttachmentState.dstColorBlendFactor = vk::BlendFactor::eOne_MINUS_SRC_ALPHA;
		blendAttachmentState.colorBlendOp = vk::BlendOp::eAdd;
		blendAttachmentState.srcAlphaBlendFactor = vk::BlendFactor::eOne;
		blendAttachmentState.dstAlphaBlendFactor = vk::BlendFactor::eZero;
		blendAttachmentState.alphaBlendOp = vk::BlendOp::eAdd;
		blendAttachmentState.colorWriteMask = vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG | vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA;

		vk::PipelineColorBlendStateCreateInfo colorBlendState =
			vks::initializers::pipelineColorBlendStateCreateInfo(
				1, 
				&blendAttachmentState);

		vk::PipelineDepthStencilStateCreateInfo depthStencilState =
			vks::initializers::pipelineDepthStencilStateCreateInfo(
				VK_FALSE,
				VK_TRUE,
				vk::CompareOp::eLessOrEqual);

		vk::PipelineViewportStateCreateInfo viewportState =
			vks::initializers::pipelineViewportStateCreateInfo(1, 1, 0);

		vk::PipelineMultisampleStateCreateInfo multisampleState =
			vks::initializers::pipelineMultisampleStateCreateInfo(
				vk::SampleCountFlagBits::e1,
				0);

		std::vector<vk::DynamicState> dynamicStateEnables = {
			vk::DynamicState::eViewport,
			vk::DynamicState::eScissor
		};
		vk::PipelineDynamicStateCreateInfo dynamicState =
			vks::initializers::pipelineDynamicStateCreateInfo(
				dynamicStateEnables.data(),
				dynamicStateEnables.size(),
				0);

		// Load shaders
		std::array<vk::PipelineShaderStageCreateInfo,2> shaderStages;

		shaderStages[0] = loadShader(getAssetPath() + "shaders/distancefieldfonts/sdf.vert.spv", vk::ShaderStageFlagBits::eVertex);
		shaderStages[1] = loadShader(getAssetPath() + "shaders/distancefieldfonts/sdf.frag.spv", vk::ShaderStageFlagBits::eFragment);

		vk::GraphicsPipelineCreateInfo pipelineCreateInfo =
			vks::initializers::pipelineCreateInfo(
				pipelineLayout,
				renderPass,
				0);

		pipelineCreateInfo.pVertexInputState = &vertices.inputState;
		pipelineCreateInfo.pInputAssemblyState = &inputAssemblyState;
		pipelineCreateInfo.pRasterizationState = &rasterizationState;
		pipelineCreateInfo.pColorBlendState = &colorBlendState;
		pipelineCreateInfo.pMultisampleState = &multisampleState;
		pipelineCreateInfo.pViewportState = &viewportState;
		pipelineCreateInfo.pDepthStencilState = &depthStencilState;
		pipelineCreateInfo.pDynamicState = &dynamicState;
		pipelineCreateInfo.stageCount = shaderStages.size();
		pipelineCreateInfo.pStages = shaderStages.data();

		pipelines.sdf = device.createGraphicsPipelines(pipelineCache, pipelineCreateInfo)[0];

		// Default bitmap font rendering pipeline
		shaderStages[0] = loadShader(getAssetPath() + "shaders/distancefieldfonts/bitmap.vert.spv", vk::ShaderStageFlagBits::eVertex);
		shaderStages[1] = loadShader(getAssetPath() + "shaders/distancefieldfonts/bitmap.frag.spv", vk::ShaderStageFlagBits::eFragment);
		pipelines.bitmap = device.createGraphicsPipelines(pipelineCache, pipelineCreateInfo)[0];
	}

	// Prepare and initialize uniform buffer containing shader uniforms
	void prepareUniformBuffers()
	{
		// Vertex shader uniform buffer block
		VK_CHECK_RESULT(vulkanDevice->createBuffer(
			vk::BufferUsageFlagBits::eUniformBuffer,
			vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
			&uniformBuffers.vs,
			sizeof(uboVS)));

		// Fragment sahder uniform buffer block (Contains font rendering parameters)
		VK_CHECK_RESULT(vulkanDevice->createBuffer(
			vk::BufferUsageFlagBits::eUniformBuffer,
			vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
			&uniformBuffers.fs,
			sizeof(uboFS)));

		// Map persistent
		uniformBuffers.vs.map();
		uniformBuffers.fs.map();

		updateUniformBuffers();
		updateFontSettings();
	}

	void updateUniformBuffers()
	{
		// Vertex shader
		glm::mat4 viewMatrix = glm::mat4();
		uboVS.projection = glm::perspective(glm::radians(splitScreen ? 30.0f : 45.0f), (float)width / (float)(height * ((splitScreen) ? 0.5f : 1.0f)), 0.001f, 256.0f);
		viewMatrix = glm::translate(viewMatrix, glm::vec3(0.0f, 0.0f, splitScreen ? zoom : zoom - 2.0f));

		uboVS.model = glm::mat4();
		uboVS.model = viewMatrix * glm::translate(uboVS.model, cameraPos);
		uboVS.model = glm::rotate(uboVS.model, glm::radians(rotation.x), glm::vec3(1.0f, 0.0f, 0.0f));
		uboVS.model = glm::rotate(uboVS.model, glm::radians(rotation.y), glm::vec3(0.0f, 1.0f, 0.0f));
		uboVS.model = glm::rotate(uboVS.model, glm::radians(rotation.z), glm::vec3(0.0f, 0.0f, 1.0f));

		memcpy(uniformBuffers.vs.mapped, &uboVS, sizeof(uboVS));
	}

	void updateFontSettings()
	{
		// Fragment shader
		memcpy(uniformBuffers.fs.mapped, &uboFS, sizeof(uboFS));
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

	void prepare()
	{
		VulkanExampleBase::prepare();
		parsebmFont();
		loadAssets();
		generateText("Vulkan");
		setupVertexDescriptions();
		prepareUniformBuffers();
		setupDescriptorSetLayout();
		preparePipelines();
		setupDescriptorPool();
		setupDescriptorSet();
		buildCommandBuffers();
		prepared = true;
	}

	virtual void render()
	{
		if (!prepared)
			return;
		vkDeviceWaitIdle(device);
		draw();
		vkDeviceWaitIdle(device);
	}

	virtual void viewChanged()
	{
		updateUniformBuffers();
	}

	void toggleSplitScreen()
	{
		splitScreen = !splitScreen;
		reBuildCommandBuffers();
		updateUniformBuffers();
	}

	void toggleFontOutline()
	{
		uboFS.outline = !uboFS.outline;
		updateFontSettings();
	}

	virtual void keyPressed(uint32_t keyCode)
	{
		switch (keyCode)
		{
		case KEY_S:
		case GAMEPAD_BUTTON_X:
			toggleSplitScreen();
			break;
		case KEY_O:
		case GAMEPAD_BUTTON_A:
			toggleFontOutline();
			break;

		}
	}

	virtual void getOverlayText(VulkanTextOverlay *textOverlay)
	{
#if defined(__ANDROID__)
		textOverlay->addText("\"Button A\" to toggle outline", 5.0f, 85.0f, VulkanTextOverlay::alignLeft);
		textOverlay->addText("\"Button X\" to toggle splitscreen", 5.0f, 100.0f, VulkanTextOverlay::alignLeft);
#else
		textOverlay->addText("\"o\" to toggle outline", 5.0f, 85.0f, VulkanTextOverlay::alignLeft);
		textOverlay->addText("\"s\" to toggle splitscreen", 5.0f, 100.0f, VulkanTextOverlay::alignLeft);
#endif
	}
};

VULKAN_EXAMPLE_MAIN()