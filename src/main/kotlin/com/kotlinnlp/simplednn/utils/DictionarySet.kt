/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.utils

import com.google.common.collect.BiMap
import com.google.common.collect.HashBiMap
import com.google.common.collect.HashMultiset
import java.io.Serializable

/**
 * A dictionary containing a set of elements.
 * Elements are mapped bi-univocally to ids.
 * It provides methods to get information about elements, like their occurrences count and frequency.
 */
class DictionarySet<T> : Serializable {

  /**
   * A [DictionarySet] factory.
   */
  companion object Factory {

    /**
     * Build a [DictionarySet] containing the given [elements].
     *
     * @param elements the elements to insert into the building dictionary
     *
     * @return a new dictionary set containing the given [elements]
     */
    operator fun <T> invoke(elements: List<T>): DictionarySet<T> {

      val dictionary = DictionarySet<T>()

      elements.forEach { dictionary.add(it) }

      return  dictionary
    }
  }

  /**
   * The number of distinct element of this set.
   */
  val size: Int get() = this.elementsMultiset.elementSet().size

  /**
   * The elements multiset with adding properties (e.g. the count of insertions of an element).
   */
  private val elementsMultiset: HashMultiset<T> = HashMultiset.create()

  /**
   * The [BiMap] of elements to ids.
   */
  private val elementsBiMap: BiMap<T, Int> = HashBiMap.create()

  /**
   * @param element an element
   *
   * @return a Boolean indicating if the dictionary contains the given element
   */
  operator fun contains(element: T): Boolean = this.elementsBiMap.containsKey(element)

  /**
   * Add the given [element] to the dictionary, incrementing the count of its occurrences.
   *
   * @param element the element to add
   */
  fun add(element: T) {

    this.elementsMultiset.add(element)

    if (element !in this.elementsBiMap) {
      this.elementsBiMap.put(key = element, value = this.elementsMultiset.elementSet().size - 1)
    }
  }

  /**
   * Get the element associated to the given [id] if it exists, null otherwise.
   *
   * @param id the id of an element
   *
   * @return the element with the given [id] or null
   */
  fun getElement(id: Int): T? = this.elementsBiMap.inverse()[id]

  /**
   * @param element an element
   *
   * @return the id of the given [element] if it is present in the dictionary, null otherwise
   */
  fun getId(element: T): Int? = this.elementsBiMap[element]

  /**
   * @param element an element
   *
   * @return the occurrences count of the given [element] (0 if it is not present)
   */
  fun getCount(element: T): Int = this.elementsMultiset.count(element)

  /**
   * @param id the id of an element
   *
   * @return the occurrences count of the element with the given [id] (0 if it is not present)
   */
  fun getCount(id: Int): Int = this.elementsMultiset.count(this.getElement(id))

  /**
   * @param id the id of an element
   *
   * @return the count of the element with the given [id]  (0 if it is not present)
   */
  fun getFrequency(id: Int): Int = this.elementsMultiset.count(this.getElement(id)) / this.size

  /**
   * @return a list of the elements in the dictionary
   */
  fun getElements(): List<T> = this.elementsMultiset.elementSet().toList()

  /**
   * @return a set of the elements in the dictionary, sorted by ascending order of occurrences
   */
  fun getElementsSortedSet(): Set<T>
    = this.elementsMultiset.elementSet().sortedBy { this.getCount(it) }.toSet()

  /**
   * @return a set of the elements in the dictionary, sorted by descending order of occurrences
   */
  fun getElementsReversedSet(): Set<T>
    = this.elementsMultiset.elementSet().sortedByDescending { this.getCount(it) }.toSet()
}
