/*
* Vulkan Example - Animated gears using multiple uniform buffers
*
* See readme.md for details
*
* Copyright (C) 2016 by Sascha Willems - www.saschawillems.de
*
* This code is licensed under the MIT license (MIT) (http://opensource.org/licenses/MIT)
*/

#include "vulkangear.h"

int32_t VulkanGear::newVertex(std::vector<Vertex> *vBuffer, float x, float y, float z, const glm::vec3& normal)
{
	Vertex v(glm::vec3(x, y, z), normal, color);
	vBuffer->push_back(v);
	return static_cast<int32_t>(vBuffer->size()) - 1;
}

void VulkanGear::newFace(std::vector<uint32_t> *iBuffer, int a, int b, int c)
{
	iBuffer->push_back(a);
	iBuffer->push_back(b);
	iBuffer->push_back(c);
}

VulkanGear::~VulkanGear()
{
	// Clean up vulkan resources
	uniformBuffer.destroy();
	vertexBuffer.destroy();
	indexBuffer.destroy();
}

void VulkanGear::generate(GearInfo *gearinfo, vk::Queue queue)
{
	this->color = gearinfo->color;
	this->pos = gearinfo->pos;
	this->rotOffset = gearinfo->rotOffset;
	this->rotSpeed = gearinfo->rotSpeed;

	std::vector<Vertex> vBuffer;
	std::vector<uint32_t> iBuffer;

	int i;
	float r0, r1, r2;
	float ta, da;
	float u1, v1, u2, v2, len;
	float cos_ta, cos_ta_1da, cos_ta_2da, cos_ta_3da, cos_ta_4da;
	float sin_ta, sin_ta_1da, sin_ta_2da, sin_ta_3da, sin_ta_4da;
	int32_t ix0, ix1, ix2, ix3, ix4, ix5;

	r0 = gearinfo->innerRadius;
	r1 = gearinfo->outerRadius - gearinfo->toothDepth / 2.0f;
	r2 = gearinfo->outerRadius + gearinfo->toothDepth / 2.0f;
	da = 2.0f * M_PI / gearinfo->numTeeth / 4.0f;

	glm::vec3 normal;

	for (i = 0; i < gearinfo->numTeeth; i++)
	{
		ta = i * 2.0f * M_PI / gearinfo->numTeeth;

		cos_ta = cos(ta);
		cos_ta_1da = cos(ta + da);
		cos_ta_2da = cos(ta + 2.0f * da);
		cos_ta_3da = cos(ta + 3.0f * da);
		cos_ta_4da = cos(ta + 4.0f * da);
		sin_ta = sin(ta);
		sin_ta_1da = sin(ta + da);
		sin_ta_2da = sin(ta + 2.0f * da);
		sin_ta_3da = sin(ta + 3.0f * da);
		sin_ta_4da = sin(ta + 4.0f * da);

		u1 = r2 * cos_ta_1da - r1 * cos_ta;
		v1 = r2 * sin_ta_1da - r1 * sin_ta;
		len = sqrt(u1 * u1 + v1 * v1);
		u1 /= len;
		v1 /= len;
		u2 = r1 * cos_ta_3da - r2 * cos_ta_2da;
		v2 = r1 * sin_ta_3da - r2 * sin_ta_2da;

		// front face
		normal = glm::vec3(0.0f, 0.0f, 1.0f);
		ix0 = newVertex(&vBuffer, r0 * cos_ta, r0 * sin_ta, gearinfo->width * 0.5f, normal);
		ix1 = newVertex(&vBuffer, r1 * cos_ta, r1 * sin_ta, gearinfo->width * 0.5f, normal);
		ix2 = newVertex(&vBuffer, r0 * cos_ta, r0 * sin_ta, gearinfo->width * 0.5f, normal);
		ix3 = newVertex(&vBuffer, r1 * cos_ta_3da, r1 * sin_ta_3da, gearinfo->width * 0.5f, normal);
		ix4 = newVertex(&vBuffer, r0 * cos_ta_4da, r0 * sin_ta_4da, gearinfo->width * 0.5f, normal);
		ix5 = newVertex(&vBuffer, r1 * cos_ta_4da, r1 * sin_ta_4da, gearinfo->width * 0.5f, normal);
		newFace(&iBuffer, ix0, ix1, ix2);
		newFace(&iBuffer, ix1, ix3, ix2);
		newFace(&iBuffer, ix2, ix3, ix4);
		newFace(&iBuffer, ix3, ix5, ix4);

		// front sides of teeth
		normal = glm::vec3(0.0f, 0.0f, 1.0f);
		ix0 = newVertex(&vBuffer, r1 * cos_ta, r1 * sin_ta, gearinfo->width * 0.5f, normal);
		ix1 = newVertex(&vBuffer, r2 * cos_ta_1da, r2 * sin_ta_1da, gearinfo->width * 0.5f, normal);
		ix2 = newVertex(&vBuffer, r1 * cos_ta_3da, r1 * sin_ta_3da, gearinfo->width * 0.5f, normal);
		ix3 = newVertex(&vBuffer, r2 * cos_ta_2da, r2 * sin_ta_2da, gearinfo->width * 0.5f, normal);
		newFace(&iBuffer, ix0, ix1, ix2);
		newFace(&iBuffer, ix1, ix3, ix2);

		// back face 
		normal = glm::vec3(0.0f, 0.0f, -1.0f);
		ix0 = newVertex(&vBuffer, r1 * cos_ta, r1 * sin_ta, -gearinfo->width * 0.5f, normal);
		ix1 = newVertex(&vBuffer, r0 * cos_ta, r0 * sin_ta, -gearinfo->width * 0.5f, normal);
		ix2 = newVertex(&vBuffer, r1 * cos_ta_3da, r1 * sin_ta_3da, -gearinfo->width * 0.5f, normal);
		ix3 = newVertex(&vBuffer, r0 * cos_ta, r0 * sin_ta, -gearinfo->width * 0.5f, normal);
		ix4 = newVertex(&vBuffer, r1 * cos_ta_4da, r1 * sin_ta_4da, -gearinfo->width * 0.5f, normal);
		ix5 = newVertex(&vBuffer, r0 * cos_ta_4da, r0 * sin_ta_4da, -gearinfo->width * 0.5f, normal);
		newFace(&iBuffer, ix0, ix1, ix2);
		newFace(&iBuffer, ix1, ix3, ix2);
		newFace(&iBuffer, ix2, ix3, ix4);
		newFace(&iBuffer, ix3, ix5, ix4);

		// back sides of teeth 
		normal = glm::vec3(0.0f, 0.0f, -1.0f);
		ix0 = newVertex(&vBuffer, r1 * cos_ta_3da, r1 * sin_ta_3da, -gearinfo->width * 0.5f, normal);
		ix1 = newVertex(&vBuffer, r2 * cos_ta_2da, r2 * sin_ta_2da, -gearinfo->width * 0.5f, normal);
		ix2 = newVertex(&vBuffer, r1 * cos_ta, r1 * sin_ta, -gearinfo->width * 0.5f, normal);
		ix3 = newVertex(&vBuffer, r2 * cos_ta_1da, r2 * sin_ta_1da, -gearinfo->width * 0.5f, normal);
		newFace(&iBuffer, ix0, ix1, ix2);
		newFace(&iBuffer, ix1, ix3, ix2);

		// draw outward faces of teeth 
		normal = glm::vec3(v1, -u1, 0.0f);
		ix0 = newVertex(&vBuffer, r1 * cos_ta, r1 * sin_ta, gearinfo->width * 0.5f, normal);
		ix1 = newVertex(&vBuffer, r1 * cos_ta, r1 * sin_ta, -gearinfo->width * 0.5f, normal);
		ix2 = newVertex(&vBuffer, r2 * cos_ta_1da, r2 * sin_ta_1da, gearinfo->width * 0.5f, normal);
		ix3 = newVertex(&vBuffer, r2 * cos_ta_1da, r2 * sin_ta_1da, -gearinfo->width * 0.5f, normal);
		newFace(&iBuffer, ix0, ix1, ix2);
		newFace(&iBuffer, ix1, ix3, ix2);

		normal = glm::vec3(cos_ta, sin_ta, 0.0f);
		ix0 = newVertex(&vBuffer, r2 * cos_ta_1da, r2 * sin_ta_1da, gearinfo->width * 0.5f, normal);
		ix1 = newVertex(&vBuffer, r2 * cos_ta_1da, r2 * sin_ta_1da, -gearinfo->width * 0.5f, normal);
		ix2 = newVertex(&vBuffer, r2 * cos_ta_2da, r2 * sin_ta_2da, gearinfo->width * 0.5f, normal);
		ix3 = newVertex(&vBuffer, r2 * cos_ta_2da, r2 * sin_ta_2da, -gearinfo->width * 0.5f, normal);
		newFace(&iBuffer, ix0, ix1, ix2);
		newFace(&iBuffer, ix1, ix3, ix2);

		normal = glm::vec3(v2, -u2, 0.0f);
		ix0 = newVertex(&vBuffer, r2 * cos_ta_2da, r2 * sin_ta_2da, gearinfo->width * 0.5f, normal);
		ix1 = newVertex(&vBuffer, r2 * cos_ta_2da, r2 * sin_ta_2da, -gearinfo->width * 0.5f, normal);
		ix2 = newVertex(&vBuffer, r1 * cos_ta_3da, r1 * sin_ta_3da, gearinfo->width * 0.5f, normal);
		ix3 = newVertex(&vBuffer, r1 * cos_ta_3da, r1 * sin_ta_3da, -gearinfo->width * 0.5f, normal);
		newFace(&iBuffer, ix0, ix1, ix2);
		newFace(&iBuffer, ix1, ix3, ix2);

		normal = glm::vec3(cos_ta, sin_ta, 0.0f);
		ix0 = newVertex(&vBuffer, r1 * cos_ta_3da, r1 * sin_ta_3da, gearinfo->width * 0.5f, normal);
		ix1 = newVertex(&vBuffer, r1 * cos_ta_3da, r1 * sin_ta_3da, -gearinfo->width * 0.5f, normal);
		ix2 = newVertex(&vBuffer, r1 * cos_ta_4da, r1 * sin_ta_4da, gearinfo->width * 0.5f, normal);
		ix3 = newVertex(&vBuffer, r1 * cos_ta_4da, r1 * sin_ta_4da, -gearinfo->width * 0.5f, normal);
		newFace(&iBuffer, ix0, ix1, ix2);
		newFace(&iBuffer, ix1, ix3, ix2);

		// draw inside radius cylinder 
		ix0 = newVertex(&vBuffer, r0 * cos_ta, r0 * sin_ta, -gearinfo->width * 0.5f, glm::vec3(-cos_ta, -sin_ta, 0.0f));
		ix1 = newVertex(&vBuffer, r0 * cos_ta, r0 * sin_ta, gearinfo->width * 0.5f, glm::vec3(-cos_ta, -sin_ta, 0.0f));
		ix2 = newVertex(&vBuffer, r0 * cos_ta_4da, r0 * sin_ta_4da, -gearinfo->width * 0.5f, glm::vec3(-cos_ta_4da, -sin_ta_4da, 0.0f));
		ix3 = newVertex(&vBuffer, r0 * cos_ta_4da, r0 * sin_ta_4da, gearinfo->width * 0.5f, glm::vec3(-cos_ta_4da, -sin_ta_4da, 0.0f));
		newFace(&iBuffer, ix0, ix1, ix2);
		newFace(&iBuffer, ix1, ix3, ix2);
	}

	size_t vertexBufferSize = vBuffer.size() * sizeof(Vertex);
	size_t indexBufferSize = iBuffer.size() * sizeof(uint32_t);

	bool useStaging = true;

	if (useStaging)
	{
		vks::Buffer vertexStaging, indexStaging;

		// Create staging buffers
		// Vertex data
		vulkanDevice->createBuffer(
			vk::BufferUsageFlagBits::eTransferSrc,
			vk::MemoryPropertyFlagBits::eHostVisible,
			&vertexStaging,
			vertexBufferSize,
			vBuffer.data());
		// Index data
		vulkanDevice->createBuffer(
			vk::BufferUsageFlagBits::eTransferSrc,
			vk::MemoryPropertyFlagBits::eHostVisible,
			&indexStaging,
			indexBufferSize,
			iBuffer.data());

		// Create device local buffers
		// Vertex buffer
		vulkanDevice->createBuffer(
			vk::BufferUsageFlagBits::eVertexBuffer | vk::BufferUsageFlagBits::eTransferDst,
			vk::MemoryPropertyFlagBits::eDeviceLocal,
			&vertexBuffer,
			vertexBufferSize);
		// Index buffer
		vulkanDevice->createBuffer(
			vk::BufferUsageFlagBits::eIndexBuffer | vk::BufferUsageFlagBits::eTransferDst,
			vk::MemoryPropertyFlagBits::eDeviceLocal,
			&indexBuffer,
			indexBufferSize);

		// Copy from staging buffers
		vk::CommandBuffer copyCmd = vulkanDevice->createCommandBuffer(vk::CommandBufferLevel::ePrimary, true);

		vk::BufferCopy copyRegion = {};

		copyRegion.size = vertexBufferSize;
		copyCmd.copyBuffer(
			vertexStaging.buffer,
			vertexBuffer.buffer,
			copyRegion);

		copyRegion.size = indexBufferSize;
		copyCmd.copyBuffer(
			indexStaging.buffer,
			indexBuffer.buffer,
			copyRegion);

		vulkanDevice->flushCommandBuffer(copyCmd, queue, true);

		vulkanDevice->logicalDevice.destroyBuffer(vertexStaging.buffer);
		vulkanDevice->logicalDevice.freeMemory(vertexStaging.memory);
		vulkanDevice->logicalDevice.destroyBuffer(indexStaging.buffer);
		vulkanDevice->logicalDevice.freeMemory(indexStaging.memory);
	}
	else
	{
		// Vertex buffer
		vulkanDevice->createBuffer(
			vk::BufferUsageFlagBits::eVertexBuffer,
			vk::MemoryPropertyFlagBits::eHostVisible,
			&vertexBuffer,
			vertexBufferSize,
			vBuffer.data());
		// Index buffer
		vulkanDevice->createBuffer(
			vk::BufferUsageFlagBits::eIndexBuffer,
			vk::MemoryPropertyFlagBits::eHostVisible,
			&indexBuffer,
			indexBufferSize,
			iBuffer.data());
	}

	indexCount = iBuffer.size();

	prepareUniformBuffer();
}

void VulkanGear::draw(vk::CommandBuffer cmdbuffer, vk::PipelineLayout pipelineLayout)
{
	std::vector<vk::DeviceSize> offsets = { 0 };
	cmdbuffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, pipelineLayout, 0, descriptorSet, nullptr);
	cmdbuffer.bindVertexBuffers(0, vertexBuffer.buffer, offsets);
	cmdbuffer.bindIndexBuffer(indexBuffer.buffer, 0, vk::IndexType::eUint32);
	cmdbuffer.drawIndexed(indexCount, 1, 0, 0, 1);
}

void VulkanGear::updateUniformBuffer(glm::mat4 perspective, glm::vec3 rotation, float zoom, float timer)
{
	ubo.projection = perspective;

	ubo.view = glm::lookAt(
		glm::vec3(0, 0, -zoom),
		glm::vec3(-1.0, -1.5, 0),
		glm::vec3(0, 1, 0)
		);
	ubo.view = glm::rotate(ubo.view, glm::radians(rotation.x), glm::vec3(1.0f, 0.0f, 0.0f));
	ubo.view = glm::rotate(ubo.view, glm::radians(rotation.y), glm::vec3(0.0f, 1.0f, 0.0f));

	ubo.model = glm::mat4();
	ubo.model = glm::translate(ubo.model, pos);
	rotation.z = (rotSpeed * timer) + rotOffset;
	ubo.model = glm::rotate(ubo.model, glm::radians(rotation.z), glm::vec3(0.0f, 0.0f, 1.0f));

	ubo.normal = glm::inverseTranspose(ubo.view * ubo.model);

	ubo.lightPos = glm::vec3(0.0f, 0.0f, 2.5f);
	ubo.lightPos.x = sin(glm::radians(timer)) * 8.0f;
	ubo.lightPos.z = cos(glm::radians(timer)) * 8.0f;

	memcpy(uniformBuffer.mapped, &ubo, sizeof(ubo));
}

void VulkanGear::setupDescriptorSet(vk::DescriptorPool pool, vk::DescriptorSetLayout descriptorSetLayout)
{
	vk::DescriptorSetAllocateInfo allocInfo =
		vks::initializers::descriptorSetAllocateInfo(
			pool,
			&descriptorSetLayout,
			1);

	descriptorSet = vulkanDevice->logicalDevice.allocateDescriptorSets(allocInfo)[0];

	// Binding 0 : Vertex shader uniform buffer
	vk::WriteDescriptorSet writeDescriptorSet =
		vks::initializers::writeDescriptorSet(
			descriptorSet,
			vk::DescriptorType::eUniformBuffer,
			0,
			&uniformBuffer.descriptor);

	vulkanDevice->logicalDevice.updateDescriptorSets(writeDescriptorSet, nullptr);
}

void VulkanGear::prepareUniformBuffer()
{
	vulkanDevice->createBuffer(
		vk::BufferUsageFlagBits::eUniformBuffer,
		vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
		&uniformBuffer,
		sizeof(ubo));
	// Map persistent
	uniformBuffer.map();
}
