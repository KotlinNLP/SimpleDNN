/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.deeplearning.attentionnetwork.han

/**
 * The HierarchyItem defines a generic item of the hierarchy which represent the input of an [HAN].
 *
 * An item could be a list of [HierarchySequence]s if it represents the lowest level of the hierarchy (the input of the
 * [HAN]) or a list of [HierarchyLevel]s if it represent a higher level.
 */
interface HierarchyItem
