/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package neuralnetwork.utils

/**
 *
 */
object SerializedNetwork {

  val byteArray = byteArrayOf(
    -84, -19, 0, 5, 115, 114, 0, 56, 99, 111, 109, 46, 107, 111, 116, 108, 105, 110, 110, 108,
    112, 46, 115, 105, 109, 112, 108, 101, 100, 110, 110, 46, 99, 111, 114, 101, 46, 110, 101, 117,
    114, 97, 108, 110, 101, 116, 119, 111, 114, 107, 46, 78, 101, 117, 114, 97, 108, 78, 101, 116,
    119, 111, 114, 107, 0, 0, 0, 0, 0, 0, 0, 1, 2, 0, 4, 90, 0, 11, 115, 112,
    97, 114, 115, 101, 73, 110, 112, 117, 116, 76, 0, 9, 105, 110, 112, 117, 116, 84, 121, 112,
    101, 116, 0, 53, 76, 99, 111, 109, 47, 107, 111, 116, 108, 105, 110, 110, 108, 112, 47, 115,
    105, 109, 112, 108, 101, 100, 110, 110, 47, 99, 111, 114, 101, 47, 108, 97, 121, 101, 114, 115,
    47, 76, 97, 121, 101, 114, 84, 121, 112, 101, 36, 73, 110, 112, 117, 116, 59, 76, 0, 19,
    108, 97, 121, 101, 114, 115, 67, 111, 110, 102, 105, 103, 117, 114, 97, 116, 105, 111, 110, 116,
    0, 16, 76, 106, 97, 118, 97, 47, 117, 116, 105, 108, 47, 76, 105, 115, 116, 59, 76, 0,
    5, 109, 111, 100, 101, 108, 116, 0, 62, 76, 99, 111, 109, 47, 107, 111, 116, 108, 105, 110,
    110, 108, 112, 47, 115, 105, 109, 112, 108, 101, 100, 110, 110, 47, 99, 111, 114, 101, 47, 110,
    101, 117, 114, 97, 108, 110, 101, 116, 119, 111, 114, 107, 47, 78, 101, 116, 119, 111, 114, 107,
    80, 97, 114, 97, 109, 101, 116, 101, 114, 115, 59, 120, 112, 0, 126, 114, 0, 51, 99, 111,
    109, 46, 107, 111, 116, 108, 105, 110, 110, 108, 112, 46, 115, 105, 109, 112, 108, 101, 100, 110,
    110, 46, 99, 111, 114, 101, 46, 108, 97, 121, 101, 114, 115, 46, 76, 97, 121, 101, 114, 84,
    121, 112, 101, 36, 73, 110, 112, 117, 116, 0, 0, 0, 0, 0, 0, 0, 0, 18, 0, 0,
    120, 114, 0, 14, 106, 97, 118, 97, 46, 108, 97, 110, 103, 46, 69, 110, 117, 109, 0, 0,
    0, 0, 0, 0, 0, 0, 18, 0, 0, 120, 112, 116, 0, 5, 68, 101, 110, 115, 101, 115,
    114, 0, 19, 106, 97, 118, 97, 46, 117, 116, 105, 108, 46, 65, 114, 114, 97, 121, 76, 105,
    115, 116, 120, -127, -46, 29, -103, -57, 97, -99, 3, 0, 1, 73, 0, 4, 115, 105, 122, 101,
    120, 112, 0, 0, 0, 2, 119, 4, 0, 0, 0, 2, 115, 114, 0, 54, 99, 111, 109, 46,
    107, 111, 116, 108, 105, 110, 110, 108, 112, 46, 115, 105, 109, 112, 108, 101, 100, 110, 110, 46,
    99, 111, 114, 101, 46, 108, 97, 121, 101, 114, 115, 46, 76, 97, 121, 101, 114, 67, 111, 110,
    102, 105, 103, 117, 114, 97, 116, 105, 111, 110, 0, 0, 0, 0, 0, 0, 0, 1, 2, 0,
    5, 68, 0, 7, 100, 114, 111, 112, 111, 117, 116, 73, 0, 4, 115, 105, 122, 101, 76, 0,
    18, 97, 99, 116, 105, 118, 97, 116, 105, 111, 110, 70, 117, 110, 99, 116, 105, 111, 110, 116,
    0, 77, 76, 99, 111, 109, 47, 107, 111, 116, 108, 105, 110, 110, 108, 112, 47, 115, 105, 109,
    112, 108, 101, 100, 110, 110, 47, 99, 111, 114, 101, 47, 102, 117, 110, 99, 116, 105, 111, 110,
    97, 108, 105, 116, 105, 101, 115, 47, 97, 99, 116, 105, 118, 97, 116, 105, 111, 110, 115, 47,
    65, 99, 116, 105, 118, 97, 116, 105, 111, 110, 70, 117, 110, 99, 116, 105, 111, 110, 59, 76,
    0, 14, 99, 111, 110, 110, 101, 99, 116, 105, 111, 110, 84, 121, 112, 101, 116, 0, 58, 76,
    99, 111, 109, 47, 107, 111, 116, 108, 105, 110, 110, 108, 112, 47, 115, 105, 109, 112, 108, 101,
    100, 110, 110, 47, 99, 111, 114, 101, 47, 108, 97, 121, 101, 114, 115, 47, 76, 97, 121, 101,
    114, 84, 121, 112, 101, 36, 67, 111, 110, 110, 101, 99, 116, 105, 111, 110, 59, 76, 0, 9,
    105, 110, 112, 117, 116, 84, 121, 112, 101, 113, 0, 126, 0, 1, 120, 112, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 3, 112, 112, 113, 0, 126, 0, 7, 115, 113, 0, 126, 0,
    11, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 112, 126, 114, 0, 56, 99, 111,
    109, 46, 107, 111, 116, 108, 105, 110, 110, 108, 112, 46, 115, 105, 109, 112, 108, 101, 100, 110,
    110, 46, 99, 111, 114, 101, 46, 108, 97, 121, 101, 114, 115, 46, 76, 97, 121, 101, 114, 84,
    121, 112, 101, 36, 67, 111, 110, 110, 101, 99, 116, 105, 111, 110, 0, 0, 0, 0, 0, 0,
    0, 0, 18, 0, 0, 120, 113, 0, 126, 0, 6, 116, 0, 11, 70, 101, 101, 100, 102, 111,
    114, 119, 97, 114, 100, 113, 0, 126, 0, 7, 120, 115, 114, 0, 60, 99, 111, 109, 46, 107,
    111, 116, 108, 105, 110, 110, 108, 112, 46, 115, 105, 109, 112, 108, 101, 100, 110, 110, 46, 99,
    111, 114, 101, 46, 110, 101, 117, 114, 97, 108, 110, 101, 116, 119, 111, 114, 107, 46, 78, 101,
    116, 119, 111, 114, 107, 80, 97, 114, 97, 109, 101, 116, 101, 114, 115, 0, 0, 0, 0, 0,
    0, 0, 1, 2, 0, 4, 90, 0, 11, 115, 112, 97, 114, 115, 101, 73, 110, 112, 117, 116,
    76, 0, 19, 108, 97, 121, 101, 114, 115, 67, 111, 110, 102, 105, 103, 117, 114, 97, 116, 105,
    111, 110, 113, 0, 126, 0, 2, 91, 0, 10, 112, 97, 114, 97, 109, 115, 76, 105, 115, 116,
    116, 0, 53, 91, 76, 99, 111, 109, 47, 107, 111, 116, 108, 105, 110, 110, 108, 112, 47, 115,
    105, 109, 112, 108, 101, 100, 110, 110, 47, 99, 111, 114, 101, 47, 97, 114, 114, 97, 121, 115,
    47, 85, 112, 100, 97, 116, 97, 98, 108, 101, 65, 114, 114, 97, 121, 59, 91, 0, 14, 112,
    97, 114, 97, 109, 115, 80, 101, 114, 76, 97, 121, 101, 114, 116, 0, 54, 91, 76, 99, 111,
    109, 47, 107, 111, 116, 108, 105, 110, 110, 108, 112, 47, 115, 105, 109, 112, 108, 101, 100, 110,
    110, 47, 99, 111, 114, 101, 47, 108, 97, 121, 101, 114, 115, 47, 76, 97, 121, 101, 114, 80,
    97, 114, 97, 109, 101, 116, 101, 114, 115, 59, 120, 114, 0, 53, 99, 111, 109, 46, 107, 111,
    116, 108, 105, 110, 110, 108, 112, 46, 115, 105, 109, 112, 108, 101, 100, 110, 110, 46, 99, 111,
    114, 101, 46, 111, 112, 116, 105, 109, 105, 122, 101, 114, 46, 73, 116, 101, 114, 97, 98, 108,
    101, 80, 97, 114, 97, 109, 115, 0, 0, 0, 0, 0, 0, 0, 1, 2, 0, 0, 120, 112,
    0, 113, 0, 126, 0, 10, 117, 114, 0, 53, 91, 76, 99, 111, 109, 46, 107, 111, 116, 108,
    105, 110, 110, 108, 112, 46, 115, 105, 109, 112, 108, 101, 100, 110, 110, 46, 99, 111, 114, 101,
    46, 97, 114, 114, 97, 121, 115, 46, 85, 112, 100, 97, 116, 97, 98, 108, 101, 65, 114, 114,
    97, 121, 59, 85, -22, 74, 103, -87, -46, -74, -78, 2, 0, 0, 120, 112, 0, 0, 0, 2,
    115, 114, 0, 55, 99, 111, 109, 46, 107, 111, 116, 108, 105, 110, 110, 108, 112, 46, 115, 105,
    109, 112, 108, 101, 100, 110, 110, 46, 99, 111, 114, 101, 46, 97, 114, 114, 97, 121, 115, 46,
    85, 112, 100, 97, 116, 97, 98, 108, 101, 68, 101, 110, 115, 101, 65, 114, 114, 97, 121, 0,
    0, 0, 0, 0, 0, 0, 1, 2, 0, 1, 76, 0, 6, 118, 97, 108, 117, 101, 115, 116,
    0, 63, 76, 99, 111, 109, 47, 107, 111, 116, 108, 105, 110, 110, 108, 112, 47, 115, 105, 109,
    112, 108, 101, 100, 110, 110, 47, 115, 105, 109, 112, 108, 101, 109, 97, 116, 104, 47, 110, 100,
    97, 114, 114, 97, 121, 47, 100, 101, 110, 115, 101, 47, 68, 101, 110, 115, 101, 78, 68, 65,
    114, 114, 97, 121, 59, 120, 114, 0, 50, 99, 111, 109, 46, 107, 111, 116, 108, 105, 110, 110,
    108, 112, 46, 115, 105, 109, 112, 108, 101, 100, 110, 110, 46, 99, 111, 114, 101, 46, 97, 114,
    114, 97, 121, 115, 46, 85, 112, 100, 97, 116, 97, 98, 108, 101, 65, 114, 114, 97, 121, 0,
    0, 0, 0, 0, 0, 0, 1, 2, 0, 2, 76, 0, 23, 117, 112, 100, 97, 116, 101, 114,
    83, 117, 112, 112, 111, 114, 116, 83, 116, 114, 117, 99, 116, 117, 114, 101, 116, 0, 84, 76,
    99, 111, 109, 47, 107, 111, 116, 108, 105, 110, 110, 108, 112, 47, 115, 105, 109, 112, 108, 101,
    100, 110, 110, 47, 99, 111, 114, 101, 47, 102, 117, 110, 99, 116, 105, 111, 110, 97, 108, 105,
    116, 105, 101, 115, 47, 117, 112, 100, 97, 116, 101, 109, 101, 116, 104, 111, 100, 115, 47, 85,
    112, 100, 97, 116, 101, 114, 83, 117, 112, 112, 111, 114, 116, 83, 116, 114, 117, 99, 116, 117,
    114, 101, 59, 76, 0, 6, 118, 97, 108, 117, 101, 115, 116, 0, 52, 76, 99, 111, 109, 47,
    107, 111, 116, 108, 105, 110, 110, 108, 112, 47, 115, 105, 109, 112, 108, 101, 100, 110, 110, 47,
    115, 105, 109, 112, 108, 101, 109, 97, 116, 104, 47, 110, 100, 97, 114, 114, 97, 121, 47, 78,
    68, 65, 114, 114, 97, 121, 59, 120, 112, 112, 115, 114, 0, 61, 99, 111, 109, 46, 107, 111,
    116, 108, 105, 110, 110, 108, 112, 46, 115, 105, 109, 112, 108, 101, 100, 110, 110, 46, 115, 105,
    109, 112, 108, 101, 109, 97, 116, 104, 46, 110, 100, 97, 114, 114, 97, 121, 46, 100, 101, 110,
    115, 101, 46, 68, 101, 110, 115, 101, 78, 68, 65, 114, 114, 97, 121, 0, 0, 0, 0, 0,
    0, 0, 1, 2, 0, 2, 76, 0, 7, 102, 97, 99, 116, 111, 114, 121, 116, 0, 70, 76,
    99, 111, 109, 47, 107, 111, 116, 108, 105, 110, 110, 108, 112, 47, 115, 105, 109, 112, 108, 101,
    100, 110, 110, 47, 115, 105, 109, 112, 108, 101, 109, 97, 116, 104, 47, 110, 100, 97, 114, 114,
    97, 121, 47, 100, 101, 110, 115, 101, 47, 68, 101, 110, 115, 101, 78, 68, 65, 114, 114, 97,
    121, 70, 97, 99, 116, 111, 114, 121, 59, 76, 0, 7, 115, 116, 111, 114, 97, 103, 101, 116,
    0, 24, 76, 111, 114, 103, 47, 106, 98, 108, 97, 115, 47, 68, 111, 117, 98, 108, 101, 77,
    97, 116, 114, 105, 120, 59, 120, 112, 115, 114, 0, 68, 99, 111, 109, 46, 107, 111, 116, 108,
    105, 110, 110, 108, 112, 46, 115, 105, 109, 112, 108, 101, 100, 110, 110, 46, 115, 105, 109, 112,
    108, 101, 109, 97, 116, 104, 46, 110, 100, 97, 114, 114, 97, 121, 46, 100, 101, 110, 115, 101,
    46, 68, 101, 110, 115, 101, 78, 68, 65, 114, 114, 97, 121, 70, 97, 99, 116, 111, 114, 121,
    0, 0, 0, 0, 0, 0, 0, 1, 2, 0, 0, 120, 112, 115, 114, 0, 22, 111, 114, 103,
    46, 106, 98, 108, 97, 115, 46, 68, 111, 117, 98, 108, 101, 77, 97, 116, 114, 105, 120, -18,
    -87, -87, 63, 50, 8, 0, 44, 3, 0, 4, 73, 0, 7, 99, 111, 108, 117, 109, 110, 115,
    73, 0, 6, 108, 101, 110, 103, 116, 104, 73, 0, 4, 114, 111, 119, 115, 91, 0, 4, 100,
    97, 116, 97, 116, 0, 2, 91, 68, 120, 112, 0, 0, 0, 3, 0, 0, 0, 15, 0, 0,
    0, 5, 117, 114, 0, 2, 91, 68, 62, -90, -116, 20, -85, 99, 90, 30, 2, 0, 0, 120,
    112, 0, 0, 0, 15, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 120, 113, 0, 126, 0, 35, 115, 113, 0, 126, 0, 26, 112, 115, 113,
    0, 126, 0, 32, 113, 0, 126, 0, 37, 115, 113, 0, 126, 0, 38, 0, 0, 0, 1, 0,
    0, 0, 5, 0, 0, 0, 5, 117, 113, 0, 126, 0, 41, 0, 0, 0, 5, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 120, 113, 0,
    126, 0, 44, 117, 114, 0, 54, 91, 76, 99, 111, 109, 46, 107, 111, 116, 108, 105, 110, 110,
    108, 112, 46, 115, 105, 109, 112, 108, 101, 100, 110, 110, 46, 99, 111, 114, 101, 46, 108, 97,
    121, 101, 114, 115, 46, 76, 97, 121, 101, 114, 80, 97, 114, 97, 109, 101, 116, 101, 114, 115,
    59, -99, 66, -74, -15, -67, 76, -47, -128, 2, 0, 0, 120, 112, 0, 0, 0, 1, 115, 114,
    0, 74, 99, 111, 109, 46, 107, 111, 116, 108, 105, 110, 110, 108, 112, 46, 115, 105, 109, 112,
    108, 101, 100, 110, 110, 46, 99, 111, 114, 101, 46, 108, 97, 121, 101, 114, 115, 46, 102, 101,
    101, 100, 102, 111, 114, 119, 97, 114, 100, 46, 70, 101, 101, 100, 102, 111, 114, 119, 97, 114,
    100, 76, 97, 121, 101, 114, 80, 97, 114, 97, 109, 101, 116, 101, 114, 115, 0, 0, 0, 0,
    0, 0, 0, 1, 2, 0, 3, 90, 0, 11, 115, 112, 97, 114, 115, 101, 73, 110, 112, 117,
    116, 91, 0, 10, 112, 97, 114, 97, 109, 115, 76, 105, 115, 116, 113, 0, 126, 0, 20, 76,
    0, 4, 117, 110, 105, 116, 116, 0, 52, 76, 99, 111, 109, 47, 107, 111, 116, 108, 105, 110,
    110, 108, 112, 47, 115, 105, 109, 112, 108, 101, 100, 110, 110, 47, 99, 111, 114, 101, 47, 108,
    97, 121, 101, 114, 115, 47, 80, 97, 114, 97, 109, 101, 116, 101, 114, 115, 85, 110, 105, 116,
    59, 120, 114, 0, 51, 99, 111, 109, 46, 107, 111, 116, 108, 105, 110, 110, 108, 112, 46, 115,
    105, 109, 112, 108, 101, 100, 110, 110, 46, 99, 111, 114, 101, 46, 108, 97, 121, 101, 114, 115,
    46, 76, 97, 121, 101, 114, 80, 97, 114, 97, 109, 101, 116, 101, 114, 115, 0, 0, 0, 0,
    0, 0, 0, 1, 2, 0, 2, 73, 0, 9, 105, 110, 112, 117, 116, 83, 105, 122, 101, 73,
    0, 10, 111, 117, 116, 112, 117, 116, 83, 105, 122, 101, 120, 113, 0, 126, 0, 22, 0, 0,
    0, 3, 0, 0, 0, 5, 0, 117, 113, 0, 126, 0, 24, 0, 0, 0, 2, 113, 0, 126,
    0, 31, 113, 0, 126, 0, 43, 115, 114, 0, 50, 99, 111, 109, 46, 107, 111, 116, 108, 105,
    110, 110, 108, 112, 46, 115, 105, 109, 112, 108, 101, 100, 110, 110, 46, 99, 111, 114, 101, 46,
    108, 97, 121, 101, 114, 115, 46, 80, 97, 114, 97, 109, 101, 116, 101, 114, 115, 85, 110, 105,
    116, 0, 0, 0, 0, 0, 0, 0, 1, 2, 0, 5, 73, 0, 9, 105, 110, 112, 117, 116,
    83, 105, 122, 101, 73, 0, 10, 111, 117, 116, 112, 117, 116, 83, 105, 122, 101, 90, 0, 11,
    115, 112, 97, 114, 115, 101, 73, 110, 112, 117, 116, 76, 0, 6, 98, 105, 97, 115, 101, 115,
    116, 0, 57, 76, 99, 111, 109, 47, 107, 111, 116, 108, 105, 110, 110, 108, 112, 47, 115, 105,
    109, 112, 108, 101, 100, 110, 110, 47, 99, 111, 114, 101, 47, 97, 114, 114, 97, 121, 115, 47,
    85, 112, 100, 97, 116, 97, 98, 108, 101, 68, 101, 110, 115, 101, 65, 114, 114, 97, 121, 59,
    76, 0, 7, 119, 101, 105, 103, 104, 116, 115, 116, 0, 52, 76, 99, 111, 109, 47, 107, 111,
    116, 108, 105, 110, 110, 108, 112, 47, 115, 105, 109, 112, 108, 101, 100, 110, 110, 47, 99, 111,
    114, 101, 47, 97, 114, 114, 97, 121, 115, 47, 85, 112, 100, 97, 116, 97, 98, 108, 101, 65,
    114, 114, 97, 121, 59, 120, 112, 0, 0, 0, 3, 0, 0, 0, 5, 0, 113, 0, 126, 0,
    43, 113, 0, 126, 0, 31
  )
}
