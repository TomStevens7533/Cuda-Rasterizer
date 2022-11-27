#include "InteropFrameBuffer.h"
#include "glew.h"
#include <assert.h>
InteropGLFrameBuffer::InteropGLFrameBuffer(const FrameBufferSpecific& spec) : m_FrameBufferSpecifics{spec}
{
	Invalidate();
}

InteropGLFrameBuffer::~InteropGLFrameBuffer()
{
	glDeleteFramebuffers(1, &m_RendererID);
	glDeleteTextures(1, &m_ColorAttached);
	glDeleteTextures(1, &m_DepthAttachment);
}

void InteropGLFrameBuffer::Invalidate()
{
	if (m_RendererID) {
		glDeleteFramebuffers(1, &m_RendererID);
		glDeleteTextures(1, &m_ColorAttached);
		glDeleteTextures(1, &m_DepthAttachment);
	}

	glCreateFramebuffers(1, &m_RendererID);
	glBindFramebuffer(GL_FRAMEBUFFER, m_RendererID);

	//Create color texture to render to
	glCreateTextures(GL_TEXTURE_2D, 1, &m_ColorAttached);
	glBindTexture(GL_TEXTURE_2D, m_ColorAttached);

	//parameters for this texture
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, m_FrameBufferSpecifics.Width, m_FrameBufferSpecifics.Height, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);


	//attach color to framebuffer
	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, m_ColorAttached, 0);

	//create depthbuffer image to render to
	glCreateTextures(GL_TEXTURE_2D, 1, &m_DepthAttachment);
	glBindTexture(GL_TEXTURE_2D, m_DepthAttachment);
	//glTexStorage2D(GL_TEXTURE_2D, 1, GL_DEPTH_STENCIL_ATTACHMENT, m_FrameBufferSpecifics.Width, m_FrameBufferSpecifics.Height);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH24_STENCIL8, m_FrameBufferSpecifics.Width, m_FrameBufferSpecifics.Height, 0, GL_DEPTH_STENCIL,
		GL_UNSIGNED_INT_24_8, NULL);

	//attach depth to framebuffer
	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_STENCIL_ATTACHMENT, GL_TEXTURE_2D, m_DepthAttachment, 0);


	assert(glCheckFramebufferStatus(GL_FRAMEBUFFER) == GL_FRAMEBUFFER_COMPLETE);

	//unbind
	glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

void InteropGLFrameBuffer::Bind()
{
	glBindFramebuffer(GL_FRAMEBUFFER, m_RendererID);
	glViewport(0, 0, m_FrameBufferSpecifics.Width, m_FrameBufferSpecifics.Height);
}

void InteropGLFrameBuffer::UnBind()
{
	glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

void InteropGLFrameBuffer::Resize(uint32_t width, uint32_t height)
{
	m_FrameBufferSpecifics.Width = width;
	m_FrameBufferSpecifics.Height = height;
	Invalidate();
}

