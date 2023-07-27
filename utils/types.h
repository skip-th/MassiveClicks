/** Type definitions.
 *
 * types.h:
 *  - Defines commonly used types, templates, and aliases used throughout
 *    the project.
 */

#ifndef TYPES_H
#define TYPES_H

#include <vector>
#include "../data/search.cuh"

// Type alias for the layout of devices in the cluster. The first dimension
// represents the node ID, and the second dimension represents the device ID.
// The value at each index is a pointer to the device.
template<typename T>
using DeviceLayout = std::vector<std::vector<T>>; // Node ID -> Device ID -> Pointer.

// LocalPartitions is a vector of tuples, where each tuple represents a
// a training set, a test set, and the number of query-document pairs assigned
// to a devices. The number of tuples in the vector is equal to the number of
// devices on the node.
using UnassignedSet = std::vector<SERP_Hst>;
using TrainSet = UnassignedSet;
using TestSet = UnassignedSet;
using Partition = std::tuple<TrainSet, TestSet, int>;
using LocalPartitions = std::vector<Partition>;

#endif // TYPES_H