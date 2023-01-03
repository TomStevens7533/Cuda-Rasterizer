#include <array>
#include "Block.h"

const std::array<float, 12> xFace2{
	0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1,
};
const std::array<float, 12> frontFace{
	0, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1,
};

const std::array<float, 12> backFace{
	1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0,
};

const std::array<float, 12> leftFace{
	0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0,
};

const std::array<float, 12> rightFace{
	1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 1, 1,
};

const std::array<float, 12> topFace{
	0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0,
};
const std::array<float, 12> bottomFace{
	0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1
};

Chunk::Chunk()
{
	GenerateFace(Faces1::FRONT);
	GenerateFace(Faces1::BACK);
	GenerateFace(Faces1::BOT);
	GenerateFace(Faces1::TOP);
	GenerateFace(Faces1::LEFT);
	GenerateFace(Faces1::RIGHT);

}

void Chunk::GenerateFace(Faces1 dir)
{
	const std::array<float, 12>* blockFace;
	std::vector<int> indices;
	switch (dir)
	{
	case Faces1::TOP:
		blockFace = &topFace;
		break;
	case Faces1::BOT:
		blockFace = &bottomFace;
		break;
	case Faces1::LEFT:
		blockFace = &leftFace;
		break;
	case Faces1::RIGHT:
		blockFace = &rightFace;
		break;
	case Faces1::FRONT:
		blockFace = &frontFace;
		break;
	case Faces1::BACK:
		blockFace = &backFace;
		break;
	default:
		break;
	}

	for (size_t i = 0; i < blockFace->size(); i += 3)
	{
		glm::vec3 Vertex1Pos = glm::vec3{ (*blockFace)[i], (*blockFace)[i + 1], (*blockFace)[i + 2] };
		m_Vertices.push_back(Vertex1Pos);
	}

	indices = { m_IndicesIndex, m_IndicesIndex + 1, m_IndicesIndex + 2, m_IndicesIndex + 2, m_IndicesIndex + 3, m_IndicesIndex };
	m_IndicesIndex += 4;
	m_Indices.insert(m_Indices.end(), indices.begin(), indices.end());

}

std::vector<int>& Chunk::GetIndices()
{
	return m_Indices;
}

std::vector<glm::vec3>& Chunk::GetVertices()
{
	return m_Vertices;
}