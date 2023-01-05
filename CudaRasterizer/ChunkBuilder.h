#pragma once
#include "glm.hpp"
#include <memory>
#include <vector>
#include "BlockType.h"	

#define  CHUNKSIZE_X 64
#define  CHUNKSIZE_Z 64
#define  CHUNKSIZE_Y 64

#define  MAX_DEPTH 4
enum class Faces
{
	TOP, BOT, LEFT, RIGHT, FRONT, BACK

};
struct SVOBaseNode {
	virtual ~SVOBaseNode() = default;
};
struct SVOInnerNode final : public SVOBaseNode{
	SVOInnerNode() = default;
	~SVOInnerNode() {
		for (size_t i = 0; i < 8; i++)
			delete children[i];
	}
	// Pointers to this node's children (if it is an internal node)
	SVOBaseNode* children[2][2][2]; //X Y Z
	SVOInnerNode* pParentNode = nullptr;
};
struct SVOLeafNode final : public SVOBaseNode{
	// Data for the voxel represented by this node
	glm::vec3 data;
	BlockTypes blockID;
	SVOInnerNode* pParentNode = nullptr;
};
class ChunkMesh final
{
public:
	ChunkMesh();
	~ChunkMesh();
	void TraverseSVO();
	std::vector<glm::vec3>& GetVertices();
	std::vector<int>& GetIndices();


private:
	void FillSVO();
	void FillSVONode(SVOBaseNode* childToFill, int depth, int resolution, int xPos, int yPos, int zPos);
	bool GetTerrainData();
	void GenerateFace(Faces dir, glm::vec3 position);
	void TraverseSVONode(SVOBaseNode* pNode, int depth);
	void CheckGenerationOfFace(Faces dir, SVOLeafNode* currLeafnode);

private:
	SVOInnerNode* m_StarterNode = nullptr;
	std::vector<int>m_Indices;
	std::vector<glm::vec3> m_Vertices;
	int m_IndicesIndex{ 0 };
	int m_blockdetected{ 0 };
	int m_Facedetected{ 0 };


};
