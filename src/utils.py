def pretty_print(text):
	"""Prints text with a topbar and bottom bar."""
	print("===============================================")
	print(text)
	print("===============================================\n")


def remove_all(lst, chars):
	"""Removes all occurrences of the characters in chars from list."""
	for char in chars:
		while char in lst:
			lst.remove(char)
	return lst
