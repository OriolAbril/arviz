(function() {
  var fn = function() {
    
    (function(root) {
      function now() {
        return new Date();
      }
    
      var force = false;
    
      if (typeof root._bokeh_onload_callbacks === "undefined" || force === true) {
        root._bokeh_onload_callbacks = [];
        root._bokeh_is_loading = undefined;
      }
    
      
      
    
      var element = document.getElementById("ac65d1f2-10ba-4e77-b13f-f9432ca4cc64");
        if (element == null) {
          console.warn("Bokeh: autoload.js configured with elementid 'ac65d1f2-10ba-4e77-b13f-f9432ca4cc64' but no matching script tag was found.")
        }
      
    
      function run_callbacks() {
        try {
          root._bokeh_onload_callbacks.forEach(function(callback) {
            if (callback != null)
              callback();
          });
        } finally {
          delete root._bokeh_onload_callbacks
        }
        console.debug("Bokeh: all callbacks have finished");
      }
    
      function load_libs(css_urls, js_urls, callback) {
        if (css_urls == null) css_urls = [];
        if (js_urls == null) js_urls = [];
    
        root._bokeh_onload_callbacks.push(callback);
        if (root._bokeh_is_loading > 0) {
          console.debug("Bokeh: BokehJS is being loaded, scheduling callback at", now());
          return null;
        }
        if (js_urls == null || js_urls.length === 0) {
          run_callbacks();
          return null;
        }
        console.debug("Bokeh: BokehJS not loaded, scheduling load and callback at", now());
        root._bokeh_is_loading = css_urls.length + js_urls.length;
    
        function on_load() {
          root._bokeh_is_loading--;
          if (root._bokeh_is_loading === 0) {
            console.debug("Bokeh: all BokehJS libraries/stylesheets loaded");
            run_callbacks()
          }
        }
    
        function on_error() {
          console.error("failed to load " + url);
        }
    
        for (var i = 0; i < css_urls.length; i++) {
          var url = css_urls[i];
          const element = document.createElement("link");
          element.onload = on_load;
          element.onerror = on_error;
          element.rel = "stylesheet";
          element.type = "text/css";
          element.href = url;
          console.debug("Bokeh: injecting link tag for BokehJS stylesheet: ", url);
          document.body.appendChild(element);
        }
    
        const hashes = {"https://cdn.bokeh.org/bokeh/release/bokeh-2.2.1.min.js": "qkRvDQVAIfzsJo40iRBbxt6sttt0hv4lh74DG7OK4MCHv4C5oohXYoHUM5W11uqS", "https://cdn.bokeh.org/bokeh/release/bokeh-widgets-2.2.1.min.js": "Sb7Mr06a9TNlet/GEBeKaf5xH3eb6AlCzwjtU82wNPyDrnfoiVl26qnvlKjmcAd+", "https://cdn.bokeh.org/bokeh/release/bokeh-tables-2.2.1.min.js": "HaJ15vgfmcfRtB4c4YBOI4f1MUujukqInOWVqZJZZGK7Q+ivud0OKGSTn/Vm2iso"};
    
        for (var i = 0; i < js_urls.length; i++) {
          var url = js_urls[i];
          var element = document.createElement('script');
          element.onload = on_load;
          element.onerror = on_error;
          element.async = false;
          element.src = url;
          if (url in hashes) {
            element.crossOrigin = "anonymous";
            element.integrity = "sha384-" + hashes[url];
          }
          console.debug("Bokeh: injecting script tag for BokehJS library: ", url);
          document.head.appendChild(element);
        }
      };
    
      function inject_raw_css(css) {
        const element = document.createElement("style");
        element.appendChild(document.createTextNode(css));
        document.body.appendChild(element);
      }
    
      
      var js_urls = ["https://cdn.bokeh.org/bokeh/release/bokeh-2.2.1.min.js", "https://cdn.bokeh.org/bokeh/release/bokeh-widgets-2.2.1.min.js", "https://cdn.bokeh.org/bokeh/release/bokeh-tables-2.2.1.min.js"];
      var css_urls = [];
      
    
      var inline_js = [
        function(Bokeh) {
          Bokeh.set_log_level("info");
        },
        
        function(Bokeh) {
          (function() {
            var fn = function() {
              Bokeh.safely(function() {
                (function(root) {
                  function embed_document(root) {
                    
                  var docs_json = '{"80c12eb1-36bd-4156-9e17-da4a62e31f90":{"roots":{"references":[{"attributes":{"formatter":{"id":"3801"},"ticker":{"id":"3753"}},"id":"3752","type":"LinearAxis"},{"attributes":{},"id":"3746","type":"LinearScale"},{"attributes":{},"id":"3778","type":"BasicTickFormatter"},{"attributes":{},"id":"3761","type":"HelpTool"},{"attributes":{},"id":"3749","type":"BasicTicker"},{"attributes":{"axis":{"id":"3748"},"ticker":null},"id":"3751","type":"Grid"},{"attributes":{"data":{"x":{"__ndarray__":"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","dtype":"float64","order":"little","shape":[512]},"y":{"__ndarray__":"vuu5i/6ngD/Rg0eBCquAP7FBGDSkqoA/Ynz0kBKygD/EhS4Q7bqAP8iSjNNexYA/K/HjJpnRgD/Guazh0t+AP1SpTqpH8IA/o16WvNT4gD8Jbd0T6hGBP+W2tAvyOoE/5o5UiRxcgT+qSp9FaHuBPyyXU4lCpYE/+WacmnfTgT9yAYrz6QyCP7rdr7/YRYI/evonazeEgj9uZpI/qMKCP1IvGpp/DYM/CJLFNWpegz9kqWySl7WDP44gZhEsE4Q/F3qQJ0B3hD9eTael3+GEP/LDMRqWWYU/Wu/m8s/YhT/+DPVL71mGP/VIVgAG6YY/C8S9I/B4hz/hj1IHqxWIP/OpAvf8v4g/vAYukhxliT/x745E6xCKP8jpoWGdyYo/G9u4pGaIiz/kIDNvdEaMP3r1sz+ND40/95tVMrndjT/r89ZoFaqOP0/bc5TOeY8/ROyF/DMmkD9B7aNO95OQPxt3X3/n/5A//qBjaJdskT+Pe71ux9mRPzr9oxJ/SpI/MBY3Or67kj8ZY/tLFCqTP7x3xVkdmJM/TyQh+OsIlD9RlP5sRnaUPxue+ETT4pQ/M/H/bT1YlT/QZBDaRcSVPwYrIJNIL5Y/azVi6aGZlj/dWb7vPwOXP4ySwk8vcpc/q86jyuLglz8FP5bPMkyYP7dwcnkHt5g/IeOERXshmT/Mke3P846ZP/4BQ7Ny85k/akVZpO1Xmj/W7dh3RL+aP/SeFXKVKZs/6LljmjyUmz+Xqfm29gWcP+5bStpveZw/85nhM3DonD8/Z0ydqGKdP8dligOr1p0/SBYyiOdJnj/fn2MQM8KeP/Xyy1z4PJ8/8ms9XADBnz+13Uo83SKgP5Dtn65/ZaA/vMdg++6poD8Uh5VrSPCgPzMHcFKoOKE/u3df/ymDoT+n6ODzF9ihP0DYW2fSJqI/pMuu9BWAoj8WWoqbydmiP7ZQhS9LNaM/EPJE4s6Voz9fL1JEYv2jP94vbjiLZKQ/xijd7FXRpD/7WJHm60ClP6TSGbRCs6U/5wb93o4rpj+N9xlYiqWmP+bQTO8RKac/t7QUnUSwpz9meRjGpUGoP+w/+c600qg/+Jw+IEBpqT8KMl7BqAWqP4xbBCS3qao/sJYDPJZVqz898s8ysAmsPw2WvVsuw6w/6yldnqWDrT/mOSlwVEauP9TKUOTIEq8/0wKTwYvprz8vUumMbmKwP/Ck9t2e1LA/mt+8I7pJsT8vEwPfRcOxP7D+zJ3pPrI/yF2lj6W/sj+IIyLwUkOzPzW5K3zczbM/Kxn3v9lZtD92POX8+Oq0P7RDsHgve7U/v8m2utYRtj97Rfx9qKu2P2lQnx75Src/lZzp+R3rtz+1rtQgso+4P+JY6AqINbk/98S7SinduT8KYV8sq4m6P5RV8JKCNbs/0vMxjxzkuz97puXrcpS8Py1SdDlhRb0/UiFdcTL4vT9vup01+au+P5tM5wV8YL8/beIK5foLwD8EdaqNpGfAP5F26zldxMA/qYb7fYsgwT8PJ4dionvBPzIHc3QL2cE/ODPSG601wj9bQYZk8pDCP6MlZXY/7MI/fNqwXGZGwz8GGLOAcaDDPxt38Z0i+8M/kaDmdJBUxD/Hp0N9za3EPwMbtE89BcU/vODFYUtcxT9NEIk/zrPFP34nxFO0CsY/M4q+ZPxgxj8hI0WWerXGP27jzuzeCMc/zOQp84ldxz/wOPMASK/HP+x3S4ouAsg/GpsdY0NTyD/0vVSeDaLIPzwauYjq8cg/DDy+c3BByT+KvnPZVJDJP02apHXT38k/QvopSmEsyj+Bx2EQUnnKP5fYreA8xco/m5GKyAIQyz8y1DyxX1zLPwqIGqiYpss/eML6nyPwyz+JZSR3lTjMP+YnpWDMf8w/bcXsp/jIzD+JFtwTqg/NP7MBkgKvVc0/SGNDAMebzT/rGU7WXeDNP8tOloMzJM4/J+GYBq5nzj+oHnVs+6nOPzoBs+k2684/Rg4oJbgrzz/MXM6iI2zPP2eXWfk+rM8/ou75beTqzz8Au+JrbRTQP4UuuMPtMtA/lWFdVPNS0D9hQyCqSXHQP2dWJ9rCjtA/h1wP0CWs0D/Se5yPScrQP1ZxOQng59A/LLwNmuQE0T8m4UY7wSHRP7Wc/C4dPtE/gykLfmha0T8wkUGc7XbRPyLPdXxFk9E/7j1UHOGv0T/G9bHTo8zRP+rjL0QX6tE/NiUO6k8H0j987X/5uCTSP/WQbJU1QtI/YkrDqwFg0j88wzKeFX7SP91N6tnYnNI/AMkPelq70j/6LRY9k9vSP9OZ8osU+9I/rAx/tKka0z9qvPXNRjzTP2OzQXrcXNM/mjjZlBl+0z9PqXlCtp/TP3Fuuw9DwdM/V5iB1rzi0z+nXQOgbgXUPxrhRmH0J9Q/4n0UCjdL1D/B79dMk27UPyqN96IuktQ//df/Qy+11D8YcJ2XRNjUP47CiKU0+9Q/4l2becQd1T9ZkoBTA0DVPzcCQxxaYtU/mU9YOpyE1T9WY7RsAKfVP6QkgWnwx9U/AOvGJKvn1T8wxrQLPwfWPyuVP5PWJdY/qC68XatD1j/880hCrWDWP8qkjYrse9Y/t40SfUmW1j8BXtqEgq/WP+mGnJa5x9Y/4FUSilDe1j+Sp8wRZPTWP80bKmWgCNc/I1Cvm2Qb1z+MMdqH7CvXP7i4SxvoO9c/acJNwG9L1z+2xuIQ/1jXP/at1dhDZdc/mkb2COZv1z9hl4/8XnnXP3tiN5Yigtc/UNF+j6mJ1z+8V/46MJDXP2V9NdPsldc/vD4u89ma1z8uACsWbJ/XP+oxe3Grotc/iVa5Tc+l1z+G0KqKKKnXP8zSCaRJq9c/eXV6l2Ws1z+l+BCiSK3XP1zlIQNOrdc/vAO81OKs1z/yUcJstKzXP8ODS59xrNc/3g5Wp22s1z+YAU/+ZKvXP40btZcZq9c/q7W0VLqr1z+LeVKJiavXPxmbTo+iq9c/MYh1Rjir1z+HVBGkUKvXP/dhCJ3Hqtc/2ZWB/xGq1z9UdtrME6nXP3FKz+Wup9c//F7AFbql1z9IcHiOL6PXP1Y25E29n9c/tCmr4puc1z+d1GkTipjXPwTk+B21ktc/pFuHrOGM1z/Xnd/q04XXP7FzTNpGfdc/ngRhCyl01z8EdXcIsWnXPzPAdS7pXNc/k8EGdxNP1z/XNr1/nT7XPy7+0E5uLtc/WErcRZsb1z9E9zZ8QgfXP8z+YUTo8dY/ZHbHuE3Z1j8HoqPL57/WPwganHmJpNY/EKXLuPaH1j8PGdMWMGnWPy3Bsou8R9Y/zpGlYWol1j9oDwffcwHWPzegJ31R29U/fMltwMWz1T8gaxnSFYvVP1rVqthlYNU/qU17mn001T8x/LWX4wbVPwLqBIw52dQ/1VDSPxuq1D9h7rdQfHnUP7huzANdSNQ/SsBHH3UW1D+4D1vGsOPTP4nSSoxNr9M/+KW9XER80z866r1n/kjTP8qgX3f5FNM/FsltYfrf0j83jTnXOKvSPzKsNbiFdtI/9cFsPvRB0j8kcpwMlw3SP9bYEQFT2dE/Les0Xk6l0T+2ALXW8nHRPyEIiWv3PdE/liZT5ooK0T8noYhkTtfQP8BK9KjfpNA/XxspaLFy0D/5mnm9/kDQP5uduRaZD9A/burO8Be+zz9n5ZodQV3PP39gck8q/s4/be1LNOudzj8+dZKkiEHOP63Caqxa5M0/YteQ8QGIzT/I8k4gly3NPxlEsNJk1cw/c1OmQtp9zD+Yg5O3ByjMP1HVym/F0ss/XMKrULp9yz+QvxbCDirLPymV2O2i1so/o50vpwSFyj9Mu2XHTzPKP5MhJF3h5Mk/0B+/JYKVyT+fcz0t20fJP1Mo97xu+sg/dJx3qASvyD+6Wfy+P2PIP5h4ux6oF8g/zNxDYGfMxz+O6yZYSoPHP9AKk7WAPMc/m3bj8o70xj/bna3Mn63GP8MyolXwZ8Y/VxPBRZQhxj9Neztj0dzFPygBGEgGmMU/LIfTQ35TxT+fyXeH+w/FP7va70+vzMQ/fLs7JmKIxD+eL8HJWkXEP4wJB4BiAsQ/icL0eay/wz8XqnUrX3zDP6yekeS7NsM/zbY0cG30wj9xZQJ0tLDCP+2hrCVdbcI/ExSYw4Uowj+GrgI4vePBPzQgHvI9nsE/anLlnCJZwT/YmoDoQRPBP3725+X3y8A/emI74ZqFwD8g7QYkUT/AP13/wqDx778/CWTiwrhevz8+3Visks++P7cUvRpAQr4/RoLQKb2zvT83qnDr/iS9P86K8MSTl7w/NQkKEH8GvD8ZYbajEni7PxXgyV0f6ro/Gvl0wM5cuj83poJnxM65P/9qHWRcQLk/YyWln3K1uD+NG36UAyu4P6jXxhBro7c/khlXw/AYtz+aI2nZE5S2P2Mq4Cz5ELY/qT3ndSeRtT/MIWbGSxK1P+EaHoHok7Q/5JTNWmQYtD+FGJFGZqCzP/n3EaYRKbM/o/9b8sW0sj9f9jCwnEKyP83KrchO07E/kKP87fBjsT9aYxWckfqwP3nQRDK1kbA/ZnZTyr4osD+PONckXIivPylaKNE2wK4/2R8x4mkArj9b19+LvT+tP6lzWOdYhKw/2o1jmHPMqz+3IZ5QFRWrP/bap9giZKo/w0jjHMm0qT/L8kn/gQipP+yNWHm9Xag/+/zFqC+6pz+kxo1xCxanPzHr5/x5dKY/NuR5B2LVpT8o6qoagTulPzOWFOb3pKQ/SJSSOu8LpD/Qb9mkgHmjPzxR9ftz6qI/UEOQT3hdoj9ma0ySycyhPwsqDF+6PqE/Cpq7j7e0oD+K7LiVlS6gP8Wu3CTtT58/EESWxPtNnj97yScgW0udP3RaSq/eS5w/QTeS+H5Ymz/4IQwh3GqaP7s4wXP0hZk/uYE7hrmmmD/k53lVMdCXP2z25xCt+ZY/GGdlh6Qslj9w3fGUTmCVP0sOQ3FkmJQ/3AaM2uvakz8+2A/baSeTP2t+vqttfZI/vYo8R7zZkT8YL7cvzTyRP6r7YBKeppA/lLUfieATkD9RjT15xRqPP5VHgvZ/E44/f4IzftMejT8xp67VETaMPx1Vxz2DUos/WzBkA7x0ij+fF+S2G66JP0wAln6j64g/wr4uaa0/iD+xFV+IBYyHP9HfB6kt7oY/qAVL9htThj/7dOkyl8aFPyPxJCrFQYU/Ax4SFlfEhD8NZ7+N/U2EP+K8wTbE2IM/WcLLov52gz8wpeCbegqDP8Gn7v/CsII/iADwf91Wgj+5pE3RAAOCP+Scfl+BuoE/Pr5vIKR2gT+rSxPllzGBP02j4EL964A/3DBbBqexgD/lEuHcg3CAPwgLu7S5TIA/2PUZioEmgD/hmLoJdQOAPzQYTocQx38/pll0BxOCfz/6bXiaAFB/P21Lb/moI38/6hWKaF7mfj8GHr+1Db1+P3ZQjXXNv34/GfAP2peafj+SeVlRAol+Pw==","dtype":"float64","order":"little","shape":[512]}},"selected":{"id":"3809"},"selection_policy":{"id":"3808"}},"id":"3786","type":"ColumnDataSource"},{"attributes":{"axis":{"id":"3752"},"dimension":1,"ticker":null},"id":"3755","type":"Grid"},{"attributes":{},"id":"3728","type":"SaveTool"},{"attributes":{},"id":"3753","type":"BasicTicker"},{"attributes":{"line_color":"#ff0000","x":{"field":"x"},"y":{"field":"y"}},"id":"3787","type":"Line"},{"attributes":{"active_drag":"auto","active_inspect":"auto","active_multi":null,"active_scroll":"auto","active_tap":"auto","tools":[{"id":"3756"},{"id":"3757"},{"id":"3758"},{"id":"3759"},{"id":"3760"},{"id":"3761"}]},"id":"3763","type":"Toolbar"},{"attributes":{"overlay":{"id":"3731"}},"id":"3727","type":"BoxZoomTool"},{"attributes":{},"id":"3780","type":"BasicTickFormatter"},{"attributes":{},"id":"3757","type":"WheelZoomTool"},{"attributes":{"children":[{"id":"3708"},{"id":"3739"}]},"id":"3791","type":"Row"},{"attributes":{},"id":"3756","type":"PanTool"},{"attributes":{"data_source":{"id":"3786"},"glyph":{"id":"3787"},"hover_glyph":null,"muted_glyph":null,"nonselection_glyph":{"id":"3788"},"selection_glyph":null,"view":{"id":"3790"}},"id":"3789","type":"GlyphRenderer"},{"attributes":{"overlay":{"id":"3762"}},"id":"3758","type":"BoxZoomTool"},{"attributes":{},"id":"3759","type":"SaveTool"},{"attributes":{"bottom":{"value":0},"fill_alpha":{"value":0.1},"fill_color":{"value":"#000000"},"left":{"field":"left"},"line_alpha":{"value":0.1},"line_color":{"value":"#000000"},"right":{"field":"right"},"top":{"field":"top"}},"id":"3772","type":"Quad"},{"attributes":{},"id":"3760","type":"ResetTool"},{"attributes":{"bottom_units":"screen","fill_alpha":0.5,"fill_color":"lightgrey","left_units":"screen","level":"overlay","line_alpha":1.0,"line_color":"black","line_dash":[4,4],"line_width":2,"right_units":"screen","top_units":"screen"},"id":"3731","type":"BoxAnnotation"},{"attributes":{"line_alpha":0.1,"line_color":"#ff0000","x":{"field":"x"},"y":{"field":"y"}},"id":"3788","type":"Line"},{"attributes":{"source":{"id":"3786"}},"id":"3790","type":"CDSView"},{"attributes":{},"id":"3713","type":"LinearScale"},{"attributes":{},"id":"3715","type":"LinearScale"},{"attributes":{"below":[{"id":"3717"}],"center":[{"id":"3720"},{"id":"3724"},{"id":"3784"}],"left":[{"id":"3721"}],"output_backend":"webgl","plot_height":500,"plot_width":500,"renderers":[{"id":"3773"}],"title":{"id":"3776"},"toolbar":{"id":"3732"},"x_range":{"id":"3709"},"x_scale":{"id":"3713"},"y_range":{"id":"3711"},"y_scale":{"id":"3715"}},"id":"3708","subtype":"Figure","type":"Plot"},{"attributes":{"formatter":{"id":"3778"},"ticker":{"id":"3722"}},"id":"3721","type":"LinearAxis"},{"attributes":{"formatter":{"id":"3780"},"ticker":{"id":"3718"}},"id":"3717","type":"LinearAxis"},{"attributes":{},"id":"3711","type":"DataRange1d"},{"attributes":{"axis":{"id":"3717"},"ticker":null},"id":"3720","type":"Grid"},{"attributes":{"text":""},"id":"3795","type":"Title"},{"attributes":{"text":""},"id":"3776","type":"Title"},{"attributes":{},"id":"3730","type":"HelpTool"},{"attributes":{},"id":"3718","type":"BasicTicker"},{"attributes":{"source":{"id":"3770"}},"id":"3774","type":"CDSView"},{"attributes":{},"id":"3808","type":"UnionRenderers"},{"attributes":{"data_source":{"id":"3770"},"glyph":{"id":"3771"},"hover_glyph":null,"muted_glyph":null,"nonselection_glyph":{"id":"3772"},"selection_glyph":null,"view":{"id":"3774"}},"id":"3773","type":"GlyphRenderer"},{"attributes":{"bottom":{"value":0},"fill_color":{"value":"#000000"},"left":{"field":"left"},"line_alpha":{"value":0},"line_color":{"value":"#000000"},"right":{"field":"right"},"top":{"field":"top"}},"id":"3771","type":"Quad"},{"attributes":{"items":[{"id":"3785"}]},"id":"3784","type":"Legend"},{"attributes":{},"id":"3809","type":"Selection"},{"attributes":{},"id":"3782","type":"UnionRenderers"},{"attributes":{"label":{"value":"Poisson"},"renderers":[{"id":"3773"}]},"id":"3785","type":"LegendItem"},{"attributes":{},"id":"3783","type":"Selection"},{"attributes":{"axis":{"id":"3721"},"dimension":1,"ticker":null},"id":"3724","type":"Grid"},{"attributes":{},"id":"3722","type":"BasicTicker"},{"attributes":{},"id":"3726","type":"WheelZoomTool"},{"attributes":{},"id":"3725","type":"PanTool"},{"attributes":{"below":[{"id":"3748"}],"center":[{"id":"3751"},{"id":"3755"}],"left":[{"id":"3752"}],"output_backend":"webgl","plot_height":500,"plot_width":500,"renderers":[{"id":"3789"}],"title":{"id":"3795"},"toolbar":{"id":"3763"},"x_range":{"id":"3740"},"x_scale":{"id":"3744"},"y_range":{"id":"3742"},"y_scale":{"id":"3746"}},"id":"3739","subtype":"Figure","type":"Plot"},{"attributes":{},"id":"3709","type":"DataRange1d"},{"attributes":{},"id":"3740","type":"DataRange1d"},{"attributes":{"active_drag":"auto","active_inspect":"auto","active_multi":null,"active_scroll":"auto","active_tap":"auto","tools":[{"id":"3725"},{"id":"3726"},{"id":"3727"},{"id":"3728"},{"id":"3729"},{"id":"3730"}]},"id":"3732","type":"Toolbar"},{"attributes":{},"id":"3729","type":"ResetTool"},{"attributes":{},"id":"3801","type":"BasicTickFormatter"},{"attributes":{"bottom_units":"screen","fill_alpha":0.5,"fill_color":"lightgrey","left_units":"screen","level":"overlay","line_alpha":1.0,"line_color":"black","line_dash":[4,4],"line_width":2,"right_units":"screen","top_units":"screen"},"id":"3762","type":"BoxAnnotation"},{"attributes":{},"id":"3744","type":"LinearScale"},{"attributes":{"formatter":{"id":"3803"},"ticker":{"id":"3749"}},"id":"3748","type":"LinearAxis"},{"attributes":{},"id":"3803","type":"BasicTickFormatter"},{"attributes":{},"id":"3742","type":"DataRange1d"},{"attributes":{"data":{"left":[0,1,2,3,4,5,6,7,8,9,10,11],"right":[1,2,3,4,5,6,7,8,9,10,11,12],"top":{"__ndarray__":"+n5qvHSTmD+TGARWDi2yP0w3iUFg5cA/NV66SQwCyz/2KFyPwvXIP+xRuB6F68E/mpmZmZmZuT8rhxbZzvezP5zEILByaKE/exSuR+F6hD/8qfHSTWJwP/yp8dJNYlA/","dtype":"float64","order":"little","shape":[12]}},"selected":{"id":"3783"},"selection_policy":{"id":"3782"}},"id":"3770","type":"ColumnDataSource"}],"root_ids":["3791"]},"title":"Bokeh Application","version":"2.2.1"}}';
                  var render_items = [{"docid":"80c12eb1-36bd-4156-9e17-da4a62e31f90","root_ids":["3791"],"roots":{"3791":"ac65d1f2-10ba-4e77-b13f-f9432ca4cc64"}}];
                  root.Bokeh.embed.embed_items(docs_json, render_items);
                
                  }
                  if (root.Bokeh !== undefined) {
                    embed_document(root);
                  } else {
                    var attempts = 0;
                    var timer = setInterval(function(root) {
                      if (root.Bokeh !== undefined) {
                        clearInterval(timer);
                        embed_document(root);
                      } else {
                        attempts++;
                        if (attempts > 100) {
                          clearInterval(timer);
                          console.log("Bokeh: ERROR: Unable to run BokehJS code because BokehJS library is missing");
                        }
                      }
                    }, 10, root)
                  }
                })(window);
              });
            };
            if (document.readyState != "loading") fn();
            else document.addEventListener("DOMContentLoaded", fn);
          })();
        },
        function(Bokeh) {
        
        
        }
      ];
    
      function run_inline_js() {
        
        for (var i = 0; i < inline_js.length; i++) {
          inline_js[i].call(root, root.Bokeh);
        }
        
      }
    
      if (root._bokeh_is_loading === 0) {
        console.debug("Bokeh: BokehJS loaded, going straight to plotting");
        run_inline_js();
      } else {
        load_libs(css_urls, js_urls, function() {
          console.debug("Bokeh: BokehJS plotting callback run at", now());
          run_inline_js();
        });
      }
    }(window));
  };
  if (document.readyState != "loading") fn();
  else document.addEventListener("DOMContentLoaded", fn);
})();