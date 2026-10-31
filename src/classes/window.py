class Window:
    SIZE = 40
    activities = {0 : 0, 1 : 1, 2 : 1, 3 : 1, 4 : 0, 5 : 0}
    def __init__(self, df, activity):
        self.data = df
        self.activity = Window.activities[int(activity)]

    def to_windows(df):
        result = []
        left = 0
        while left < len(df) - Window.SIZE:
            window = df.iloc[left : left + Window.SIZE]
            left += Window.SIZE
            if window['activity'].std() != 0.0:
                continue
            result.append(Window(window, window.iloc[0, 1]))
        return result

    def to_json(self):
        return self.data.to_json()
