from loguru import logger

logger.debug("This is Debug")
logger.info("This is INFO")


def test(num_list):
    for num in num_list:
        logger.info(f"This Number is " + num)


if __name__ == "__main__":
    l = [1, 2, 3]
    test(num_list=l)