class _MustBe:
  """ class for asserting that a dimension must have a certain value.
      the class itself is private, one should import a particular object,
      "must_be" in order to use the functionality. example code:
      `batch, chan, mustbe[32], mustbe[32] = image.shape`
      `*must_be[batch, 20, 20], chan = tens.shape` """
  def __setitem__(self, key, value):
    if isinstance(key, tuple):
      assert key == tuple(value), "must_be[%s] does not match dimension %s" % (str(key), str(value))
    else:
      assert key == value, "must_be[%d] does not match dimension %d" % (key, value)
must_be = _MustBe()

