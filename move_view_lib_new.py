import math

class view_mover():
    def __init__(self):
        self.R = 1.0
        self.e=0.0818191908426
        self.init_position(116.34469,39.9751)

    def init_position(self, Longitude, Latitude):
        self.Latitude = Latitude*math.pi/180.0
        self.Longitude = Longitude*math.pi/180.0
        self.update_Rn_Re()

    def update_Rn_Re(self):
        self.Rn=self.R*(1-(self.e**2))/(1-(self.e**2)*(math.sin(self.Latitude))**2)**1.5
        self.Re=self.R/(1-(self.e**2)*(math.sin(self.Latitude))**2)**0.5

    def move_view(self, direction, degree_per_step):

        Vn = degree_per_step / 180.0 * math.pi * math.cos(direction / 180.0 * math.pi)
        Ve = degree_per_step / 180.0 * math.pi * math.sin(direction / 180.0 * math.pi)

        self.Latitude=self.Latitude+Vn/self.Rn;
        self.Longitude=self.Longitude+Ve/(self.Re*math.cos(self.Latitude));
        self.update_Rn_Re()
        if self.Longitude < -180.0:
            self.Longitude += 360.0
        if self.Longitude > 180.0:
            self.Longitude -= 360.0
        return self.Longitude,self.Latitude
