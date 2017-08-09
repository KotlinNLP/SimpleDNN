/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.utils

/**
 * A pool of items which allows to allocate and release them when needed, without creating new ones every time.
 * E.g.: it is useful to optimize the creation of new structures every time a NeuralProcessor is needed.
 */
abstract class ItemsPool<ItemType: ItemsPool.IDItem> {

  /**
   * An interface which defines an item with an [id] property.
   */
  interface IDItem {
    val id: Int
  }

  /**
   * The size of the pool (all items).
   */
  val size: Int get() = this.pool.size

  /**
   * The number of items in use.
   */
  val usage: Int get() = this.pool.size - this.availableItems.size

  /**
   * The pool of all the created items.
   */
  private val pool = arrayListOf<ItemType>()

  /**
   * A set containing the ids of the items not in use.
   */
  private val availableItems = mutableSetOf<Int>()

  /**
   * Get an item currently not in use (and set it as in use).
   */
  fun getItem(): ItemType {

    if (availableItems.size == 0) {
      this.addItem()
    }

    return this.popAvailableItem()
  }

  /**
   * Set a item as available again.
   */
  fun releaseItem(item: ItemType) {
    this.availableItems.add(item.id)
  }

  /**
   * Set all items as available again.
   */
  fun releaseAll() {
    this.pool.forEach { this.availableItems.add(it.id) }
  }

  /**
   * Add a new item to the pool.
   */
  private fun addItem() {

    val item = this.itemFactory(id = this.pool.size)

    this.pool.add(item)
    this.availableItems.add(item.id)
  }

  /**
   * Pop the first available item removing it from the list of available ones (the pool is required to be not empty).
   *
   * @return the first available item
   */
  private fun popAvailableItem(): ItemType {

    val itemId: Int = this.availableItems.first()
    this.availableItems.remove(itemId)

    return this.pool[itemId]
  }

  /**
   * The factory of a new item.
   *
   * @param id the unique id of the item to create
   *
   * @return a new item with the given [id]
   */
  abstract fun itemFactory(id: Int): ItemType
}
