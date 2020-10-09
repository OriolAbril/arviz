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
    
      
      
    
      var element = document.getElementById("388019c4-d126-4218-8dfc-eecb7deef6c1");
        if (element == null) {
          console.warn("Bokeh: autoload.js configured with elementid '388019c4-d126-4218-8dfc-eecb7deef6c1' but no matching script tag was found.")
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
                    
                  var docs_json = '{"2730c3e0-aae8-496c-8057-f7eecf8d4909":{"roots":{"references":[{"attributes":{"fill_alpha":{"value":0.1},"fill_color":{"value":"#1f77b4"},"line_alpha":{"value":0.1},"line_color":{"value":"#1f77b4"},"x":{"field":"x"},"y":{"field":"y"}},"id":"21989","type":"Circle"},{"attributes":{},"id":"21902","type":"DataRange1d"},{"attributes":{"children":[{"id":"22027"},{"id":"22025"}]},"id":"22028","type":"Column"},{"attributes":{"data":{"rug_x":{"__ndarray__":"fV36E1z/6j89DycBWWfXP73VlTJ7YsE/w8E5M5BGxz/DwTkzkEbHP8PBOTOQRsc/w8E5M5BGxz/DwTkzkEbHP8PBOTOQRsc/w8E5M5BGxz/DwTkzkEbHP8PBOTOQRsc/w8E5M5BGxz/DwTkzkEbHP8PBOTOQRsc/w8E5M5BGxz9P6PYIJnnCP0/o9ggmecI/T+j2CCZ5wj9P6PYIJnnCP0/o9ggmecI/T+j2CCZ5wj9P6PYIJnnCP0/o9ggmecI/txiR/XzQvD+3GJH9fNC8P7cYkf180Lw/txiR/XzQvD+3GJH9fNC8P7cYkf180Lw/Oqay45Jr6D86SHRMZcflP7HThSU1z+I/VdSaCTtd6D9Hvab2ZmSwP8UA5kQ6d4M/0GULqag1oz+aI/Yi4T7rP55DEK8H/NA/qgGd6qjX1D+VqnS/h2ThP/UnuP7VN+Q/TGXHJdeQ2z8=","dtype":"float64","order":"little","shape":[43]},"rug_y":{"__ndarray__":"AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA=","dtype":"float64","order":"little","shape":[43]}},"selected":{"id":"22022"},"selection_policy":{"id":"22023"}},"id":"21996","type":"ColumnDataSource"},{"attributes":{"source":{"id":"21996"}},"id":"21998","type":"CDSView"},{"attributes":{"fill_alpha":{"value":0.1},"fill_color":{"value":"#1f77b4"},"line_alpha":{"value":0.1},"line_color":{"value":"#1f77b4"},"x":{"field":"x"},"y":{"field":"y"}},"id":"21975","type":"Circle"},{"attributes":{"data":{"x":{"__ndarray__":"mpmZmZmZqT/SjhUII+24P2wor6G8hsI/bolTv+eWyD9w6vfcEqfOP7klTv2eW9I/O1YgjLRj1T+8hvIaymvYPz23xKnfc9s/vueWOPV73j8gjLRjBcLgP2GkHSsQRuI/oryG8hrK4z/i1O+5JU7lPyPtWIEw0uY/YwXCSDtW6D+kHSsQRtrpP+U1lNdQXus/JU79nlvi7D9mZmZmZmbuPw==","dtype":"float64","order":"little","shape":[20]},"y":{"__ndarray__":"bcJe+Pxh3D88Eu5SUq3NPyL+eQAns8s/mzeOVbYJ1z+YvNPsmzTWP87j+UyletE/sIr1s8Bnzz8YN/msnr7PP8gPs0h4ec8/RNGUEkZ90z8g12riYrDUP+AlTxYjYc4/UBTOvdhAzD+AFwpyJ0DOPxDSzcUXbc4/kDNyJsikyj9AY0p3Si3PP8AWdsF70MQ/4I8dAXxLyD+grhW5nZrSPw==","dtype":"float64","order":"little","shape":[20]}},"selected":{"id":"22020"},"selection_policy":{"id":"22021"}},"id":"21987","type":"ColumnDataSource"},{"attributes":{"below":[{"id":"21946"}],"center":[{"id":"21949"},{"id":"21953"}],"left":[{"id":"21950"}],"output_backend":"webgl","plot_height":500,"plot_width":500,"renderers":[{"id":"21990"},{"id":"21992"},{"id":"21993"},{"id":"21994"},{"id":"21997"}],"title":{"id":"21999"},"toolbar":{"id":"21964"},"toolbar_location":null,"x_range":{"id":"21938"},"x_scale":{"id":"21942"},"y_range":{"id":"21940"},"y_scale":{"id":"21944"}},"id":"21937","subtype":"Figure","type":"Plot"},{"attributes":{"source":{"id":"21987"}},"id":"21991","type":"CDSView"},{"attributes":{"below":[{"id":"21910"}],"center":[{"id":"21913"},{"id":"21917"}],"left":[{"id":"21914"}],"output_backend":"webgl","plot_height":500,"plot_width":500,"renderers":[{"id":"21976"},{"id":"21978"},{"id":"21979"},{"id":"21980"},{"id":"21983"}],"title":{"id":"21985"},"toolbar":{"id":"21928"},"toolbar_location":null,"x_range":{"id":"21902"},"x_scale":{"id":"21906"},"y_range":{"id":"21904"},"y_scale":{"id":"21908"}},"id":"21901","subtype":"Figure","type":"Plot"},{"attributes":{"data_source":{"id":"21987"},"glyph":{"id":"21988"},"hover_glyph":null,"muted_glyph":null,"nonselection_glyph":{"id":"21989"},"selection_glyph":null,"view":{"id":"21991"}},"id":"21990","type":"GlyphRenderer"},{"attributes":{"toolbars":[{"id":"21928"},{"id":"21964"}],"tools":[{"id":"21918"},{"id":"21919"},{"id":"21920"},{"id":"21921"},{"id":"21922"},{"id":"21923"},{"id":"21924"},{"id":"21925"},{"id":"21954"},{"id":"21955"},{"id":"21956"},{"id":"21957"},{"id":"21958"},{"id":"21959"},{"id":"21960"},{"id":"21961"}]},"id":"22026","type":"ProxyToolbar"},{"attributes":{"angle":{"units":"rad","value":1.5707963267948966},"line_alpha":{"value":0.35},"size":{"units":"screen","value":8},"x":{"field":"rug_x"},"y":{"field":"rug_y"}},"id":"21995","type":"Dash"},{"attributes":{"axis_label":"Quantile","formatter":{"id":"22017"},"ticker":{"id":"21947"}},"id":"21946","type":"LinearAxis"},{"attributes":{"line_alpha":0.5,"line_width":1.5,"location":0.2148430013731262},"id":"21992","type":"Span"},{"attributes":{"end":1,"start":-0.05},"id":"21940","type":"DataRange1d"},{"attributes":{"line_alpha":0.5,"line_width":0.75,"location":0.15209716424958658},"id":"21993","type":"Span"},{"attributes":{"axis_label":"MCSE for quantiles","formatter":{"id":"22015"},"ticker":{"id":"21951"}},"id":"21950","type":"LinearAxis"},{"attributes":{"data_source":{"id":"21996"},"glyph":{"id":"21995"},"hover_glyph":null,"muted_glyph":null,"view":{"id":"21998"}},"id":"21997","type":"GlyphRenderer"},{"attributes":{},"id":"21942","type":"LinearScale"},{"attributes":{"text":"mu"},"id":"21999","type":"Title"},{"attributes":{},"id":"21944","type":"LinearScale"},{"attributes":{},"id":"21947","type":"BasicTicker"},{"attributes":{"line_alpha":0.7,"line_width":1.5,"location":0},"id":"21994","type":"Span"},{"attributes":{"axis":{"id":"21946"},"ticker":null},"id":"21949","type":"Grid"},{"attributes":{"toolbar":{"id":"22026"},"toolbar_location":"above"},"id":"22027","type":"ToolbarBox"},{"attributes":{"axis":{"id":"21950"},"dimension":1,"ticker":null},"id":"21953","type":"Grid"},{"attributes":{"children":[[{"id":"21901"},0,0],[{"id":"21937"},0,1]]},"id":"22025","type":"GridBox"},{"attributes":{"axis_label":"Quantile","formatter":{"id":"22005"},"ticker":{"id":"21911"}},"id":"21910","type":"LinearAxis"},{"attributes":{},"id":"21951","type":"BasicTicker"},{"attributes":{"overlay":{"id":"21962"}},"id":"21956","type":"BoxZoomTool"},{"attributes":{},"id":"21955","type":"PanTool"},{"attributes":{},"id":"21954","type":"ResetTool"},{"attributes":{},"id":"21957","type":"WheelZoomTool"},{"attributes":{},"id":"22015","type":"BasicTickFormatter"},{"attributes":{"overlay":{"id":"21963"}},"id":"21958","type":"LassoSelectTool"},{"attributes":{},"id":"21915","type":"BasicTicker"},{"attributes":{},"id":"21911","type":"BasicTicker"},{"attributes":{},"id":"21923","type":"UndoTool"},{"attributes":{},"id":"22003","type":"BasicTickFormatter"},{"attributes":{},"id":"21906","type":"LinearScale"},{"attributes":{"callback":null},"id":"21925","type":"HoverTool"},{"attributes":{"active_drag":"auto","active_inspect":"auto","active_multi":null,"active_scroll":"auto","active_tap":"auto","tools":[{"id":"21918"},{"id":"21919"},{"id":"21920"},{"id":"21921"},{"id":"21922"},{"id":"21923"},{"id":"21924"},{"id":"21925"}]},"id":"21928","type":"Toolbar"},{"attributes":{"axis":{"id":"21914"},"dimension":1,"ticker":null},"id":"21917","type":"Grid"},{"attributes":{"text":"tau"},"id":"21985","type":"Title"},{"attributes":{"overlay":{"id":"21926"}},"id":"21920","type":"BoxZoomTool"},{"attributes":{},"id":"21919","type":"PanTool"},{"attributes":{"bottom_units":"screen","fill_alpha":0.5,"fill_color":"lightgrey","left_units":"screen","level":"overlay","line_alpha":1.0,"line_color":"black","line_dash":[4,4],"line_width":2,"right_units":"screen","top_units":"screen"},"id":"21962","type":"BoxAnnotation"},{"attributes":{},"id":"21918","type":"ResetTool"},{"attributes":{},"id":"21924","type":"SaveTool"},{"attributes":{},"id":"21921","type":"WheelZoomTool"},{"attributes":{},"id":"22017","type":"BasicTickFormatter"},{"attributes":{},"id":"22011","type":"UnionRenderers"},{"attributes":{"overlay":{"id":"21927"}},"id":"21922","type":"LassoSelectTool"},{"attributes":{},"id":"21938","type":"DataRange1d"},{"attributes":{"axis_label":"MCSE for quantiles","formatter":{"id":"22003"},"ticker":{"id":"21915"}},"id":"21914","type":"LinearAxis"},{"attributes":{"active_drag":"auto","active_inspect":"auto","active_multi":null,"active_scroll":"auto","active_tap":"auto","tools":[{"id":"21954"},{"id":"21955"},{"id":"21956"},{"id":"21957"},{"id":"21958"},{"id":"21959"},{"id":"21960"},{"id":"21961"}]},"id":"21964","type":"Toolbar"},{"attributes":{"fill_alpha":0.5,"fill_color":"lightgrey","level":"overlay","line_alpha":1.0,"line_color":"black","line_dash":[4,4],"line_width":2,"xs_units":"screen","ys_units":"screen"},"id":"21963","type":"PolyAnnotation"},{"attributes":{},"id":"22021","type":"UnionRenderers"},{"attributes":{"fill_color":{"value":"#1f77b4"},"line_color":{"value":"#1f77b4"},"x":{"field":"x"},"y":{"field":"y"}},"id":"21974","type":"Circle"},{"attributes":{},"id":"21959","type":"UndoTool"},{"attributes":{},"id":"22020","type":"Selection"},{"attributes":{},"id":"21960","type":"SaveTool"},{"attributes":{"callback":null},"id":"21961","type":"HoverTool"},{"attributes":{"axis":{"id":"21910"},"ticker":null},"id":"21913","type":"Grid"},{"attributes":{"data":{"x":{"__ndarray__":"mpmZmZmZqT/SjhUII+24P2wor6G8hsI/bolTv+eWyD9w6vfcEqfOP7klTv2eW9I/O1YgjLRj1T+8hvIaymvYPz23xKnfc9s/vueWOPV73j8gjLRjBcLgP2GkHSsQRuI/oryG8hrK4z/i1O+5JU7lPyPtWIEw0uY/YwXCSDtW6D+kHSsQRtrpP+U1lNdQXus/JU79nlvi7D9mZmZmZmbuPw==","dtype":"float64","order":"little","shape":[20]},"y":{"__ndarray__":"TOY7mm62yD/upazjteTNP/DqXfy8Lcs/HHCIRHVOzT9orzckCyTOP6y8CkfbP8w/iNx/9NF5yz+IfM+LFJ/NPygi8KloJ8w/oK0q3zNVzD/YcLH58jHPP6hWLqhoMNQ/IGaMl5nu0j8AebTCVJLRP+D0hUiIM80/uNpKBpu90z/A00Fq0J3TP2iCFjW8ldY/kLBgWUSo1j8QCPgZgPnXPw==","dtype":"float64","order":"little","shape":[20]}},"selected":{"id":"22008"},"selection_policy":{"id":"22009"}},"id":"21973","type":"ColumnDataSource"},{"attributes":{},"id":"22005","type":"BasicTickFormatter"},{"attributes":{"end":1,"start":-0.05},"id":"21904","type":"DataRange1d"},{"attributes":{},"id":"22022","type":"Selection"},{"attributes":{"bottom_units":"screen","fill_alpha":0.5,"fill_color":"lightgrey","left_units":"screen","level":"overlay","line_alpha":1.0,"line_color":"black","line_dash":[4,4],"line_width":2,"right_units":"screen","top_units":"screen"},"id":"21926","type":"BoxAnnotation"},{"attributes":{},"id":"22023","type":"UnionRenderers"},{"attributes":{},"id":"22009","type":"UnionRenderers"},{"attributes":{},"id":"22008","type":"Selection"},{"attributes":{"data":{"rug_x":{"__ndarray__":"jQwCEA1Gsz9bBMLb9PjIP1Qd9Zram7E/yuyJRU9bmT/K7IlFT1uZP8rsiUVPW5k/yuyJRU9bmT/K7IlFT1uZP8rsiUVPW5k/yuyJRU9bmT/K7IlFT1uZP8rsiUVPW5k/yuyJRU9bmT/K7IlFT1uZP8rsiUVPW5k/yuyJRU9bmT/zis1V/mKlP/OKzVX+YqU/84rNVf5ipT/zis1V/mKlP/OKzVX+YqU/84rNVf5ipT/zis1V/mKlP/OKzVX+YqU/aF85wG2piz9oXznAbamLP2hfOcBtqYs/aF85wG2piz9oXznAbamLP2hfOcBtqYs/pI3yRkqEyT9c/+ob+nG6P4OiBeyjALU/mx4fY+a33D9wF2c1cbTRPy51Aws2htg/P07TEgOYwz9FOncTGHy5P/BQ+ANPucc/uRO6PYJJzj9CRCWTDYpmP+UvVnGFsrI/8JR71fNwcj8=","dtype":"float64","order":"little","shape":[43]},"rug_y":{"__ndarray__":"AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA=","dtype":"float64","order":"little","shape":[43]}},"selected":{"id":"22010"},"selection_policy":{"id":"22011"}},"id":"21982","type":"ColumnDataSource"},{"attributes":{"fill_alpha":0.5,"fill_color":"lightgrey","level":"overlay","line_alpha":1.0,"line_color":"black","line_dash":[4,4],"line_width":2,"xs_units":"screen","ys_units":"screen"},"id":"21927","type":"PolyAnnotation"},{"attributes":{"source":{"id":"21982"}},"id":"21984","type":"CDSView"},{"attributes":{"data_source":{"id":"21973"},"glyph":{"id":"21974"},"hover_glyph":null,"muted_glyph":null,"nonselection_glyph":{"id":"21975"},"selection_glyph":null,"view":{"id":"21977"}},"id":"21976","type":"GlyphRenderer"},{"attributes":{},"id":"21908","type":"LinearScale"},{"attributes":{"source":{"id":"21973"}},"id":"21977","type":"CDSView"},{"attributes":{"angle":{"units":"rad","value":1.5707963267948966},"line_alpha":{"value":0.35},"size":{"units":"screen","value":8},"x":{"field":"rug_x"},"y":{"field":"rug_y"}},"id":"21981","type":"Dash"},{"attributes":{"line_alpha":0.5,"line_width":1.5,"location":0.2515582690238702},"id":"21978","type":"Span"},{"attributes":{"line_alpha":0.5,"line_width":0.75,"location":0.1782444431478369},"id":"21979","type":"Span"},{"attributes":{},"id":"22010","type":"Selection"},{"attributes":{"line_alpha":0.7,"line_width":1.5,"location":0},"id":"21980","type":"Span"},{"attributes":{"data_source":{"id":"21982"},"glyph":{"id":"21981"},"hover_glyph":null,"muted_glyph":null,"view":{"id":"21984"}},"id":"21983","type":"GlyphRenderer"},{"attributes":{"fill_color":{"value":"#1f77b4"},"line_color":{"value":"#1f77b4"},"x":{"field":"x"},"y":{"field":"y"}},"id":"21988","type":"Circle"}],"root_ids":["22028"]},"title":"Bokeh Application","version":"2.2.1"}}';
                  var render_items = [{"docid":"2730c3e0-aae8-496c-8057-f7eecf8d4909","root_ids":["22028"],"roots":{"22028":"388019c4-d126-4218-8dfc-eecb7deef6c1"}}];
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